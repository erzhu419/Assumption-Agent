#!/usr/bin/env python3
"""Run autonomous prior research against real local SC-KG simulator outputs.

All shared/cold learning finishes before fresh evaluation. Frozen learned
programs are then executed as features by SC-KG's own ParametricGPR, with equal
new data for learned, linear, and the ordinary SC-KG quadratic basis.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import __version__
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.sc_kg_environment import SCKGEnvironment, compile_sc_kg_program


BASES = (843000013, 843100019, 843200029)
ARMS = ("shared", "cold")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def protocol(project_root):
    config = ResearchConfig(observation_budget=96, compute_budget=60000,
                            max_rounds=4, minimum_improvement=1e-7)
    return {
        "experiment": "sc_kg_autonomous_meta_prior_and_gp_consumption",
        "sc_kg_project_root": str(Path(project_root).resolve()),
        "cohorts": {f"sequence_{index + 1}": [
            {"problem": ("rzdt1", "rzdt2")[position % 2],
             "dimension": 5, "seed": base + 10007 * position, "noise": 0.0}
            for position in range(4)] for index, base in enumerate(BASES)},
        "arms": list(ARMS), "config": asdict(config),
        "evaluation": {"seed_offset": 1_000_003, "rows": 128},
        "sc_kg_gp": {"train_seed_offset": 2_000_003, "training_rows": 8,
                     "evaluation_seed_offset": 3_000_017, "evaluation_rows": 64,
                     "lambda_i": 1e-12, "prior_var": 1e6, "observation_variance": 1e-8,
                     "comparators": ["learned_program", "linear_x1", "sc_kg_default_quadratic"]},
        "learning": "Shared retains one KnowledgeLibrary per sequence; cold starts a new library per task. Both retain within-task prior revision. Task and evaluation seeds are identical across arms.",
        "evaluation_timing": "All 24 learning tasks must finish before independent measurements or GP training are requested. No evaluation observations return to the library.",
        "accounting": "Actual simulator calls are counted. Each frozen model uses 128 primary evaluation calls, 8 new GP training calls and 64 GP evaluation calls. The three GP models share identical newly acquired data, rather than reacquiring it three times.",
        "scope": "Two finite noise-free RZDT constraint families at dimension five. Public simulate outputs only; this evaluates hypothesis discovery and cross-project consumption, not optimization improvement or the current SC-KG paper method.",
    }


def learn(spec, directory):
    from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
    for cohort, tasks in spec["cohorts"].items():
        for arm in ARMS:
            path = directory / f"{cohort}_{arm}_learning.json"
            state = json.loads(path.read_text()) if path.exists() else {
                "implementation_version": __version__, "cohort": cohort, "arm": arm,
                "results": [], "library": KnowledgeLibrary().to_dict(), "status": "learning",
                "active": None}
            if state["implementation_version"] != __version__:
                raise ValueError("learning version differs; use a new output directory")
            while len(state["results"]) < len(tasks):
                position = len(state["results"])
                task = tasks[position]
                active = state.get("active")
                library = (KnowledgeLibrary.from_dict(active["library"]) if active else
                           KnowledgeLibrary.from_dict(state["library"])
                           if arm == "shared" else KnowledgeLibrary())
                environment = SCKGEnvironment(spec["sc_kg_project_root"], **task)
                config = replace(ResearchConfig(**spec["config"]), seed=task["seed"])
                def checkpoint(report):
                    state["active"] = {"result": report, "library": library.to_dict()}
                    write_json(path, state)
                result = MetaPriorResearch(config, library=library).run(environment,
                    research_state=active["result"]["research_state"] if active else None,
                    on_checkpoint=checkpoint)
                state["results"].append({"position": position, "task": task,
                                         "result": result, "actual_costs": environment.costs})
                state["library"] = library.to_dict()
                state["active"] = None
                state["status"] = "complete" if len(state["results"]) == len(tasks) else "learning"
                write_json(path, state)
                print(json.dumps({"cohort": cohort, "arm": arm, "position": position,
                                  "status": result.get("status"), "costs": result.get("costs")}), flush=True)


def _mse(program, rows, fallback):
    values = []
    for row in rows:
        prediction = program.predict(row) if program is not None else fallback
        values.append((prediction - row.target) ** 2)
    return mean(values)


def _gp_comparison(task, environment, program, settings):
    import numpy as np
    from core.gpr import ParametricGPR
    if program is None:
        return {"status": "no_program", "training_simulator_calls": 0, "evaluation_simulator_calls": 0}
    try:
        adapter = compile_sc_kg_program(environment, program)
    except ValueError as error:
        return {"status": "unsupported_program", "reason": str(error),
                "training_simulator_calls": 0, "evaluation_simulator_calls": 0}
    linear = compile_sc_kg_program(environment, Program("feature", name="x1"))
    kwargs = {key: settings[key] for key in ("lambda_i", "prior_var")}
    models = {
        "learned_program": ParametricGPR(d=task["dimension"], **kwargs,
                                           basis_map=adapter.gpr_basis_map(output_index=1)),
        "linear_x1": ParametricGPR(d=task["dimension"], **kwargs,
                                     basis_map=linear.gpr_basis_map(output_index=1)),
        "sc_kg_default_quadratic": ParametricGPR(d=task["dimension"], **kwargs,
                                                  normalize_func=adapter.normalize),
    }
    rng = np.random.default_rng(task["seed"] + settings["train_seed_offset"])
    lower, upper = adapter.int_bounds()
    training = []
    for _ in range(settings["training_rows"]):
        point = tuple(int(rng.integers(lo, hi + 1)) for lo, hi in zip(lower, upper))
        measured = float(adapter.simulate(point, rng)[1])
        training.append({"point": list(point), "observed_constraint": measured})
        for model in models.values():
            model.update(point, measured, sigma2_hat=settings["observation_variance"])
    paid_before = environment.evaluation_costs["simulator_calls"]
    rows = environment.evaluation_rows(seed=task["seed"] + settings["evaluation_seed_offset"],
                                        count=settings["evaluation_rows"])
    points = [environment.integer_point(row.features) for row in rows]
    targets = np.asarray([row.target for row in rows])
    metrics = {}
    for name, model in models.items():
        predictions = model.posterior_mean_many(points)
        metrics[name] = {"mse": float(np.mean((predictions - targets) ** 2)),
                         "max_absolute_error": float(np.max(np.abs(predictions - targets))),
                         "basis_dimension_with_intercept": model.p}
    return {"status": "evaluated", "models": metrics, "training": training,
            "training_simulator_calls": adapter.simulator_calls,
            "evaluation_simulator_calls": environment.evaluation_costs["simulator_calls"] - paid_before,
            "learned_basis_evaluations": adapter.basis.evaluations}


def evaluate(spec, directory):
    states = []
    for cohort, tasks in spec["cohorts"].items():
        for arm in ARMS:
            path = directory / f"{cohort}_{arm}_learning.json"
            state = json.loads(path.read_text())
            if state["status"] != "complete" or len(state["results"]) != len(tasks):
                raise ValueError("all shared/cold learning must finish before evaluation")
            states.append(state)
    evaluated = []
    for state in states:
        path = directory / f"{state['cohort']}_{state['arm']}_evaluation.json"
        if path.exists():
            evaluated.extend(json.loads(path.read_text())["results"])
            continue
        rows = []
        for item in state["results"]:
            task, report = item["task"], item["result"]
            environment = SCKGEnvironment(spec["sc_kg_project_root"], **task)
            program = Program.from_dict(report["program"]) if report.get("program") else None
            observations = environment.evaluation_rows(seed=task["seed"] + spec["evaluation"]["seed_offset"],
                                                        count=spec["evaluation"]["rows"])
            metrics = {"cohort": state["cohort"], "arm": state["arm"],
                       "position": item["position"], "task": task,
                       "status": report.get("status"),
                       "mse": _mse(program, observations, report.get("fallback", 0.0)),
                       "learning_costs": report.get("costs", {}),
                       "actual_learning_simulator_calls": item["actual_costs"]["simulator_calls"],
                       "primary_evaluation_simulator_calls": environment.evaluation_costs["simulator_calls"],
                       "claims": report.get("claims", []), "prior_updates": report.get("prior_updates", []),
                       "events": report.get("events", [])}
            metrics["sc_kg_gp"] = _gp_comparison(task, environment, program, spec["sc_kg_gp"])
            rows.append(metrics)
        write_json(path, {"cohort": state["cohort"], "arm": state["arm"], "results": rows})
        evaluated.extend(rows)
    summary = {"implementation_version": __version__, "protocol": spec,
               "results": evaluated, "aggregate": {}}
    for arm in ARMS:
        rows = [row for row in evaluated if row["arm"] == arm]
        gp_rows = [row["sc_kg_gp"] for row in rows if row["sc_kg_gp"]["status"] == "evaluated"]
        summary["aggregate"][arm] = {
            "task_count": len(rows), "mean_mse": mean(row["mse"] for row in rows),
            "max_mse": max(row["mse"] for row in rows),
            "learning_simulator_calls": sum(row["actual_learning_simulator_calls"] for row in rows),
            "learning_compute_cost": sum(row["learning_costs"].get("compute_cost", 0) for row in rows),
            "primary_evaluation_calls": sum(row["primary_evaluation_simulator_calls"] for row in rows),
            "gp_evaluated_tasks": len(gp_rows),
            "gp_training_calls": sum(row["training_simulator_calls"] for row in gp_rows),
            "gp_evaluation_calls": sum(row["evaluation_simulator_calls"] for row in gp_rows),
            "gp_mean_mse": {name: mean(row["models"][name]["mse"] for row in gp_rows)
                            for name in spec["sc_kg_gp"]["comparators"]} if gp_rows else {},
        }
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path,
                        default=ROOT.parents[1] / "KG_op" / "SC-OLH-KG")
    parser.add_argument("--phase", choices=("learn", "evaluate", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = protocol(args.project_root)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("the saved study protocol differs; use a new output directory")
    if not path.exists():
        write_json(path, spec)
    if args.phase in {"learn", "all"}:
        learn(spec, args.output)
    if args.phase in {"evaluate", "all"}:
        evaluate(spec, args.output)


if __name__ == "__main__":
    main()
