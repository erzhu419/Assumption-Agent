#!/usr/bin/env python3
"""Paired, offline research streams with reuse, change and return stages."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
from random import Random
import shutil
from statistics import mean
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
OUTPUT = ROOT / "artifacts/mixed_research_investment_20260920"
BASELINE = "artifacts/relation_kernel_composition_20260920/runtime/src"
ARMS = ("baseline_v0106", "research_investment", "task_reset")
FAMILIES = ("objects", "risk")
STAGES = ("cold_source", "same_structure", "changed_conditions", "return_conditions")
COSTS = ("observation_cost", "measurement_cost", "compute_cost")


def save(path, payload):
    write_json_checkpoint(path, payload, indent=None, ensure_ascii=False, allow_nan=True, separators=(',', ':'))


def checkpoint(output, cohort, family, arm, seed):
    return output / cohort / "runs" / family / arm / str(seed) / "checkpoint.json"


def prepare(output):
    path = output / "protocol.json"
    if path.exists():
        raise ValueError("preserve the declared protocol; choose a new output for a different study")
    if not (ROOT / BASELINE / "hegel_machine/cognition/unified_session.py").exists():
        raise FileNotFoundError(BASELINE)
    cohorts = {"development": [96008001], "main": list(range(96016001, 96016009))}
    tasks, scenarios = {family: {} for family in FAMILIES}, {}
    for index, seed in enumerate([*cohorts["development"], *cohorts["main"]]):
        cell = max(0, index - 1)
        rng = Random(seed + 17011)
        count, change = (2, 3)[(cell // 2) % 2], ("instrument", "mechanism")[cell % 2]
        base = {"environment": "objects", "objects": count, "readout": "focus", "variant": "exchange",
            "apparatus_seed": seed, "profile": "confirmation", "actuator_curvature": rng.uniform(.08, .20),
            "relation_radius": rng.uniform(.70, 1.15)}
        changed = {**base, **({"actuator_curvature": base["actuator_curvature"] + .18}
                             if change == "instrument" else {"readout": "event"})}
        tasks["objects"][str(seed)] = [{**item, "seed": seed + 10007 * position}
            for position, item in enumerate((base, base, changed, base))]
        tasks["risk"][str(seed)] = [{"environment": "scalar", "task_index": task,
            "profile": profile, "seed": seed + 200003 + 10007 * position}
            for position, (task, profile) in enumerate(((12, "development"), (12, "development"),
                                                       (13, "confirmation"), (12, "development")))]
        scenarios[str(seed)] = {"objects": count, "object_change": change}
    families = {
        "objects": {"research_config": {"observation_budget": 256, "compute_budget": 180000,
            "minimum_improvement": 1e-7, "max_rounds": 6, "max_history": 4, "active_probes": True},
            "measurement_budget": 3072, "loss_kind": "mse", "risk_quantile": .5},
        "risk": {"research_config": {"observation_budget": 256, "compute_budget": 160000,
            "minimum_improvement": 1e-7, "max_rounds": 4, "max_history": 4, "active_probes": False},
            "measurement_budget": 1024, "loss_kind": "pinball", "risk_quantile": .9}}
    for settings in families.values():
        settings["stream_budget"] = {"observation_cost": 4 * settings["research_config"]["observation_budget"],
            "measurement_cost": 4 * settings["measurement_budget"],
            "compute_cost": 4 * settings["research_config"]["compute_budget"]}
    save(path, {"experiment": "mixed_research_investment_streams", "fixed_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": cohorts, "tasks_by_family_seed": tasks, "scenarios": scenarios, "stages": STAGES,
        "development_runtime": "src", "families": families,
        "arms": {ARMS[0]: {"runtime": BASELINE, "version": "0.106.0"},
            ARMS[1]: {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.107.0"},
            ARMS[2]: {"runtime": str((output / "runtime/src").relative_to(ROOT)), "version": "0.107.0"}},
        "shared_laboratory_source": BASELINE + "/hegel_machine/cognition", "maximum_concurrent_processes": 4,
        "terminal": {"objects": {"length": 48, "trajectories": 4, "random_backgrounds": 2,
            "sweep_fractions": [0., .5, 1.], "mse_tolerance": 1e-6},
            "risk": {"length": 64, "trajectories": 8}, "seed_offset": 1000003},
        "budget_contract": "Each family/root is an independent four-task session with one original resource pool, "
            "equal to the sum of the four declared first-visit allowances. Ordinary session scheduling may spend "
            "unused visited-task resources on revisits; every source task, failed attempt and screening operation counts. "
            "No extra revisit allowance. Object and risk pools are separate and identical across arms.",
        "reset_contract": "The task-reset arm starts each new task with an empty KnowledgeLibrary, keeping ordinary "
            "within-task learning and the complete paid report. Cross-task offers are disabled. Previous task reports "
            "and observations are preserved, not erased or repurchased. This control removes both cross-task knowledge "
            "and the revisits it would trigger; it does not separately identify knowledge versus scheduling effects.",
        "learning_contract": "Every session starts without seeded concepts. JSON restore occurs between visits. "
            "All learning in all families and arms of a cohort must end before any terminal evaluation. Main uses "
            "frozen source only and retains all unresolved results, exceptions and resource use. No terminal tuning.",
        "evaluation_contract": "Object models face four new random trajectories plus public low/mid/high control "
            "steps at the center and two seeded random backgrounds. Risk models face eight new random trajectories "
            "at unchanged quantile .9. Use actual public laboratory acquisitions; model predictions are the actions. "
            "Terminal rows never enter session knowledge. Report object MSE/pass and risk pinball separately, "
            "including every stage and full paired stream. Fixed (9,1) action loss is exactly ten times pinball.",
        "scope": "Eight new object apparatus roots with two repeats of each 2/3-object by instrument/mechanism-change "
            "cell, and eight noise/control realizations of existing risk mechanisms. Four stages per independent "
            "family/root, not a single mixed-loss metric. The one exposed development root is not prospective evidence.",
        "formal_evidence": False})


def freeze(output):
    destination = output / "runtime/src/hegel_machine"
    if destination.exists():
        raise ValueError("preserve the frozen runtime")
    shutil.copytree(ROOT / "src/hegel_machine", destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def load(protocol, arm, cohort):
    source = ROOT / (protocol["development_runtime"] if cohort == "development" and arm != ARMS[0]
                     else protocol["arms"][arm]["runtime"])
    sys.path.insert(0, str(source))
    for name in ("structure_lab", "unified_object_lab", "unknown_task_lab"):
        canonical = "hegel_machine.cognition." + name
        spec = importlib.util.spec_from_file_location(canonical,
            ROOT / protocol["shared_laboratory_source"] / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = module
        spec.loader.exec_module(module)
    module = importlib.import_module("hegel_machine.cognition.unified_session")
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("each arm must run in a separate process")
    session_type = module.UnifiedResearchSession
    if arm == ARMS[2]:
        class TaskResetSession(session_type):
            def _offer_new_knowledge(self, *args, **kwargs):
                return None
        session_type = TaskResetSession
    def factory(item):
        arguments = {key: value for key, value in item.items() if key not in ("environment", "research_limits")}
        name = "unified_object_lab" if item["environment"] == "objects" else "unknown_task_lab"
        lab = sys.modules["hegel_machine.cognition." + name]
        return (lab.UnifiedObjectLab if name == "unified_object_lab" else lab.UnknownTaskLab)(**arguments)
    return module, session_type, factory


def learn(protocol, output, cohort, family, arm, seed, max_visits=None):
    module, session_type, make_lab = load(protocol, arm, cohort)
    path = checkpoint(output, cohort, family, arm, seed)
    settings = protocol["families"][family]
    laboratories = []
    def factory(item):
        lab = make_lab(item)
        laboratories.append(lab)
        return lab
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("study_learning_complete"):
            return
        session = session_type(state=state, lab_factory=factory)
    else:
        config = module.ResearchConfig(**settings["research_config"], seed=seed)
        session = session_type(config, tasks=protocol["tasks_by_family_seed"][family][str(seed)],
            measurement_budget=settings["measurement_budget"], lab_factory=factory)
    visits = 0
    while len(session.state["results"]) < len(STAGES) or session.state["pending_revisits"]:
        if max_visits is not None and visits >= max_visits:
            break
        first_count, revisit_count = len(session.state["results"]), len(session.state["revisits"])
        if arm == ARMS[2] and first_count < len(STAGES):
            session.library = module.KnowledgeLibrary()
        before = session.to_dict()["costs"]
        laboratories.clear()
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            state = session.run(max_tasks=1)
        except Exception:
            failed = session.to_dict()
            failed["study_failure"] = {"traceback": traceback.format_exc(), "first_visits": first_count,
                "revisits": revisit_count, "actual_new_costs": dict(sum((Counter(lab.costs) for lab in laboratories), Counter())),
                "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall}
            save(path, failed)
            raise
        actual = dict(sum((Counter(lab.costs) for lab in laboratories), Counter()))
        delta = {key: state["costs"][key] - before[key] for key in COSTS}
        for key in ("observation_cost", "measurement_cost"):
            if actual.get(key, 0) != delta[key]:
                save(path, state)
                raise RuntimeError(f"actual and reported purchases differ: {key}")
        if any(state["costs"][key] > settings["stream_budget"][key] for key in COSTS):
            save(path, state)
            raise RuntimeError("the original complete-stream resource pool was exceeded")
        fresh_task = len(state["results"]) > first_count
        position = first_count if fresh_task else (state["revisits"][-1]["task_position"]
            if len(state["revisits"]) > revisit_count else None)
        state.setdefault("study_visits", []).append({"position": position, "first_visit": fresh_task,
            "incremental_costs": delta, "actual_new_costs": actual,
            "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall})
        if fresh_task:
            report = state["results"][position]["research"]
            state.setdefault("study_initial_models", {})[str(position)] = {key: deepcopy(report[key])
                for key in ("program", "fallback", "selected_channels", "status", "costs")}
        state["study_learning_complete"] = len(state["results"]) == len(STAGES) and not state["pending_revisits"]
        save(path, state)
        print(json.dumps({"family": family, "arm": arm, "seed": seed, "position": position,
            "first_visit": fresh_task, "incremental_costs": delta, "finished": state["study_learning_complete"]}), flush=True)
        session = session_type(state=json.loads(path.read_text()), lab_factory=factory)
        visits += 1


def ensure_learning(protocol, output, cohort):
    for family in FAMILIES:
        for arm in ARMS:
            for seed in protocol["cohorts"][cohort]:
                path = checkpoint(output, cohort, family, arm, seed)
                if not path.exists() or not json.loads(path.read_text()).get("study_learning_complete"):
                    raise ValueError(f"finish all cohort learning before evaluation: {family}/{arm}/{seed}")


def evaluate(protocol, output, cohort, family, arm, seed):
    ensure_learning(protocol, output, cohort)
    _, _, factory = load(protocol, arm, cohort)
    from hegel_machine.cognition.evaluation import predict, point_loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Observation, Probe
    path = checkpoint(output, cohort, family, arm, seed)
    state = json.loads(path.read_text())
    settings = protocol["terminal"][family]
    for position in range(len(state["evaluation"]), len(STAGES)):
        task = state["tasks"][position]
        report, initial = state["results"][position]["research"], state["study_initial_models"][str(position)]
        models = {"final": report, "initial": initial}
        programs = {key: Program.from_dict(item["program"]) for key, item in models.items()}
        with factory(task) as lab:
            q = lab.spec.risk_quantile
            if q != protocol["families"][family]["risk_quantile"]:
                raise ValueError("terminal evaluation must retain the public learning loss")
            required = {tuple(item["selected_channels"]) for item in models.values()}
            if family == "objects":
                rates = {channel.name: channel.cost_per_row for channel in lab.channels}
                cover = min((tuple(view) for view in lab.measurement_views
                    if required <= {(), *map(tuple, lab.projectable_views(view))}),
                    key=lambda view: sum(rates[name] for name in view))
            else:
                cover = tuple(channel.name for channel in lab.channels
                              if any(channel.name in view for view in required))
            def score(probe):
                acquired = lab.acquire(probe, cover)
                command_history, commands = [], []
                for point, row in zip(probe.sequence, acquired.observations):
                    commands.append(Observation(dict(point), row.target, tuple(command_history)))
                    command_history.append({**point, "__target__": row.target})
                result = {"probe_id": probe.probe_id, "rows": len(acquired.observations)}
                for key, model in models.items():
                    view = tuple(model["selected_channels"])
                    rows = (commands if not view else lab.project_rows(acquired.observations, cover, view)
                            if family == "objects" else acquired.observations)
                    values = [predict(programs[key], row, model["fallback"]) for row in rows]
                    losses = [point_loss(value, row.target, q) for value, row in zip(values, rows)]
                    result[key + "_loss"] = mean(losses)
                    if family == "risk":
                        result[key + "_coverage"] = mean(row.target <= value for value, row in zip(values, rows))
                        result[key + "_action_loss"] = mean(9 * max(row.target - value, 0.) + max(value - row.target, 0.)
                                                               for value, row in zip(values, rows))
                return result
            random = []
            for index in range(settings["trajectories"]):
                rng = Random(task["seed"] + protocol["terminal"]["seed_offset"] + index * 10007)
                points = tuple({name: rng.uniform(low, high) for name, (low, high) in lab.spec.feature_bounds.items()}
                               for _ in range(settings["length"]))
                random.append(score(Probe(f"mixed_terminal_random:{index}", points)))
            stress = []
            if family == "objects":
                bounds = lab.spec.feature_bounds
                backgrounds = [{name: (low + high) / 2 for name, (low, high) in bounds.items()}]
                for index in range(settings["random_backgrounds"]):
                    rng = Random(task["seed"] + 1900033 + index * 10007)
                    backgrounds.append({name: rng.uniform(low, high) for name, (low, high) in bounds.items()})
                for index, point in enumerate(backgrounds):
                    for control, (low, high) in sorted(bounds.items()):
                        points = [dict(point) for _ in range(3)] + [
                            {**point, control: low + fraction * (high - low)} for fraction in settings["sweep_fractions"]]
                        stress.append(score(Probe(f"mixed_terminal_step:{index}:{control}", tuple([*points, point, point]))))
            unique_rows = sum(item["rows"] for item in [*random, *stress])
            evaluation = {"task_position": position, "stage": STAGES[position], "risk_quantile": q,
                "loss_kind": protocol["families"][family]["loss_kind"], "random": random, "stress": stress,
                "costs": {**lab.costs, "compute_cost": 2 * unique_rows}, "actual_channels": list(cover)}
            rates = {channel.name: channel.cost_per_row for channel in lab.channels}
            for key, model in models.items():
                random_loss = mean(row[key + "_loss"] for row in random)
                evaluation[key + "_random_loss"] = random_loss
                evaluation[key + "_loss"] = max(random_loss, mean(row[key + "_loss"] for row in stress)) if stress else random_loss
                evaluation[key + "_deployment_measurement_cost"] = unique_rows * sum(rates[name] for name in model["selected_channels"])
                if stress:
                    evaluation[key + "_stress_loss"] = mean(row[key + "_loss"] for row in stress)
                    evaluation[key + "_pass"] = evaluation[key + "_loss"] <= settings["mse_tolerance"]
                else:
                    evaluation[key + "_coverage"] = mean(row[key + "_coverage"] for row in random)
                    evaluation[key + "_action_loss"] = mean(row[key + "_action_loss"] for row in random)
        state["evaluation"].append(evaluation)
        save(path, state)


def summarize(protocol, output, cohort):
    result = {"cohort": cohort, "scope": protocol["scope"], "families": {}}
    for family in FAMILIES:
        entry = {"loss_kind": protocol["families"][family]["loss_kind"], "arms": {}, "paired": {}}
        for arm in ARMS:
            tasks, streams, costs, terminal = [], [], Counter(), Counter()
            for seed in protocol["cohorts"][cohort]:
                state = json.loads(checkpoint(output, cohort, family, arm, seed).read_text())
                if len(state["evaluation"]) != len(STAGES):
                    raise ValueError("all returned models must be evaluated, including unresolved ones")
                visits = state.get("study_visits", [])
                streams.append({"seed": seed, "loss": mean(row["final_loss"] for row in state["evaluation"]),
                    "costs": state["costs"], "screening_compute_cost": state["screening_compute_cost"],
                    "revisits": len(state["revisits"]), "cpu_seconds": sum(row["cpu_seconds"] for row in visits),
                    "wall_seconds": sum(row["wall_seconds"] for row in visits)})
                costs.update(state["costs"])
                for record, evaluation in zip(state["results"], state["evaluation"]):
                    terminal.update(evaluation["costs"])
                    report = record["research"]
                    tasks.append({"seed": seed, "position": record["task_position"], "stage": STAGES[record["task_position"]],
                        "status": report["status"], "selected_kind": report["selected_kind"],
                        "channels": report["selected_channels"], "costs": report["costs"],
                        "evaluation": {key: value for key, value in evaluation.items() if key not in ("random", "stress")},
                        "actions": dict(Counter(event["action"] for event in report["actions"]))})
            entry["arms"][arm] = {"mean_loss": mean(row["loss"] for row in streams), "streams": streams, "tasks": tasks,
                "by_stage": {stage: mean(row["evaluation"]["final_loss"] for row in tasks if row["stage"] == stage)
                             for stage in STAGES}, "learning_costs": dict(costs), "terminal_costs": dict(terminal),
                "statuses": dict(Counter(row["status"] for row in tasks)),
                "passes": sum(row["evaluation"].get("final_pass", False) for row in tasks) if family == "objects" else None,
                "cpu_seconds": sum(row["cpu_seconds"] for row in streams),
                "wall_seconds": sum(row["wall_seconds"] for row in streams)}
        learned = entry["arms"][ARMS[1]]["streams"]
        for control in (ARMS[0], ARMS[2]):
            old = {row["seed"]: row for row in entry["arms"][control]["streams"]}
            changes = {str(row["seed"]): row["loss"] - old[row["seed"]]["loss"] for row in learned}
            tolerance = protocol["terminal"]["objects"]["mse_tolerance"] if family == "objects" else 0.
            entry["paired"][control] = {"new_minus_control": changes, "mean_change": mean(changes.values()),
                "lower": sum(value < -tolerance for value in changes.values()),
                "higher": sum(value > tolerance for value in changes.values()),
                "equal": sum(abs(value) <= tolerance for value in changes.values())}
        result["families"][family] = entry
    save(output / cohort / "summary.json", result)
    digest = deepcopy(result)
    for entry in digest["families"].values():
        for data in entry["arms"].values():
            data.pop("tasks")
    save(output / cohort / "final_digest.json", digest)
    print(json.dumps(digest, ensure_ascii=False), flush=True)


def queue(protocol, output, cohort):
    jobs = [(family, arm, seed) for family in FAMILIES for arm in ARMS for seed in protocol["cohorts"][cohort]]
    history = {"cohort": cohort, "status": "running", "jobs": []}
    for phase in ("learn", "evaluate"):
        def execute(job):
            family, arm, seed = job
            path = output / cohort / "logs" / f"{phase}_{family}_{arm}_{seed}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                code = subprocess.call([sys.executable, "-B", str(Path(__file__).resolve()), phase,
                    "--output", str(output), "--cohort", cohort, "--family", family, "--arm", arm,
                    "--seed", str(seed)], stdout=stream, stderr=subprocess.STDOUT)
            return {"phase": phase, "family": family, "arm": arm, "seed": seed, "exit_code": code}
        with ThreadPoolExecutor(max_workers=protocol["maximum_concurrent_processes"]) as pool:
            pending = [pool.submit(execute, job) for job in jobs]
            for future in as_completed(pending):
                record = future.result()
                history["jobs"].append(record)
                save(output / cohort / "execution.json", history)
                print(json.dumps(record), flush=True)
        if any(item["exit_code"] for item in history["jobs"]):
            history["status"] = "failed"
            save(output / cohort / "execution.json", history)
            raise SystemExit(1)
    summarize(protocol, output, cohort)
    history["status"] = "completed"
    save(output / cohort / "execution.json", history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "learn", "evaluate", "queue", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cohort", choices=("development", "main"), default="development")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-visits", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "freeze":
        freeze(args.output)
    else:
        protocol = json.loads((args.output / "protocol.json").read_text())
        if args.mode in ("queue", "summarize"):
            globals()[args.mode](protocol, args.output, args.cohort)
        else:
            if args.family is None or args.arm is None or args.seed not in protocol["cohorts"][args.cohort]:
                parser.error("choose a family, arm and predeclared cohort seed")
            if args.mode == "learn":
                learn(protocol, args.output, args.cohort, args.family, args.arm, args.seed, args.max_visits)
            else:
                evaluate(protocol, args.output, args.cohort, args.family, args.arm, args.seed)


if __name__ == "__main__":
    main()
