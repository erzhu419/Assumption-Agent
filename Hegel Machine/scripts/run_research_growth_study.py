#!/usr/bin/env python3
"""Compare persistent and task-local research on one eight-task sequence."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys
from time import process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.growth_lab import GrowthLab, STAGES
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.models import ResearchConfig

RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")
ARMS = ("shared", "task_local")


def protocol(seeds):
    return {"experiment": "integrated_research_growth", "seeds": list(seeds),
        "stages": list(STAGES), "arms": list(ARMS),
        "research_config": asdict(ResearchConfig(observation_budget=384,
            compute_budget=200000, max_rounds=6, minimum_improvement=1e-7)),
        "measurement_budget": 1024,
        "terminal": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "acceptance": {"deterministic_mse": 1e-5, "stochastic_mean_mse": .03,
            "stochastic_variance_ratio": [.25, 4.]},
        "information_contract": "Eight tasks use one public interface without stage labels. "
            "Both arms retain within-task research; only shared retains persistent knowledge "
            "between tasks. Each task has the same resource caps. All training finishes "
            "before terminal scoring, whose latent means and noise variance are evaluator-only.",
        "external_model_calls": 0}


def evaluate(report, laboratory, evaluation, acceptance, seed):
    data = laboratory.evaluation_rows(report["selected_channels"], seed=seed, **{
        key: evaluation[key] for key in ("length", "trajectories")})
    program = Program.from_dict(report["program"])
    predictions, primitive_steps = [], 0
    for row in data["observations"]:
        context = EvaluationContext(row)
        predictions.append(context.evaluate(program))
        primitive_steps += context.primitive_steps
    mse = mean((prediction - row.target) ** 2
        for prediction, row in zip(predictions, data["observations"]))
    mean_mse = mean((prediction - row.target) ** 2
        for prediction, row in zip(predictions, data["mean_observations"]))
    variance = data["irreducible_noise_variance"]
    estimated = report["uncertainty"].get("variance", 0.)
    ratio = estimated / variance if variance else None
    noise_identified = bool(variance and report["uncertainty"]["kind"] == "stochastic"
        and acceptance["stochastic_variance_ratio"][0] <= ratio
        <= acceptance["stochastic_variance_ratio"][1])
    solved = (mean_mse <= acceptance["stochastic_mean_mse"] and noise_identified
        if variance else mse <= acceptance["deterministic_mse"])
    return {"mse": mse, "mean_mse": mean_mse, "solved": bool(solved),
        "uncertainty": report["uncertainty"], "true_noise_variance": variance,
        "noise_variance_ratio": ratio, "noise_identified": noise_identified,
        "prediction_rows": len(predictions), "evaluation_primitive_steps": primitive_steps,
        "evaluation_costs": {**data["costs"], "compute_cost": len(predictions)}}


def run_case(seed, plan):
    from hegel_machine.cognition.research_growth import GrowthResearcher

    config = ResearchConfig(**plan["research_config"])
    reports, final_states = {}, {}
    for arm in ARMS:
        state, tasks = None, []
        for stage, name in enumerate(STAGES):
            before = process_time()
            researcher = GrowthResearcher(config=replace(config, seed=seed + 10007 * stage),
                state=state if arm == "shared" else None,
                measurement_budget=plan["measurement_budget"])
            laboratory = GrowthLab(seed=seed, stage=stage)
            report = researcher.run(laboratory)
            # Persistence is exercised between real tasks, rather than only serialized at the end.
            state = json.loads(json.dumps(researcher.to_dict()))
            tasks.append({"stage": stage, "stage_name": name, "research": report,
                "training_cpu_seconds": process_time() - before})
        reports[arm] = tasks
        final_states[arm] = state
    # The same fresh controls and realizations score both arms only after their training.
    results = {}
    for arm, tasks in reports.items():
        for task in tasks:
            stage = task["stage"]
            task.update(evaluate(task["research"], GrowthLab(seed=seed, stage=stage),
                plan["terminal"], plan["acceptance"],
                seed + plan["terminal"]["seed_offset"] + 10007 * stage))
        results[arm] = {"tasks": tasks, "solved": sum(task["solved"] for task in tasks),
            "confirmed": sum(task["research"]["status"] == "confirmed" for task in tasks),
            "costs": {resource: sum(task["research"]["costs"][resource] for task in tasks)
                      for resource in RESOURCES},
            "training_cpu_seconds": sum(task["training_cpu_seconds"] for task in tasks),
            "final_state": final_states[arm]}
    difference = {resource: results["shared"]["costs"][resource]
        - results["task_local"]["costs"][resource] for resource in RESOURCES}
    return {"seed": seed, "results": results,
        "paired_effect": {"shared_minus_task_local_cost": difference,
            "solved_difference": results["shared"]["solved"] - results["task_local"]["solved"]},
        "external_model_calls": 0}


def summarize(cases, plan):
    totals = {}
    for arm in ARMS:
        rows = [case["results"][arm] for case in cases]
        totals[arm] = {"tasks": len(cases) * len(STAGES),
            "solved": sum(row["solved"] for row in rows),
            "confirmed": sum(row["confirmed"] for row in rows),
            "costs": {resource: sum(row["costs"][resource] for row in rows) for resource in RESOURCES},
            "training_cpu_seconds": sum(row["training_cpu_seconds"] for row in rows),
            "stages": [{"stage": stage, "stage_name": name,
                "solved": sum(row["tasks"][stage]["solved"] for row in rows),
                "mean_mse": mean(row["tasks"][stage]["mean_mse"] for row in rows),
                "response_mse": mean(row["tasks"][stage]["mse"] for row in rows),
                "costs": {resource: sum(row["tasks"][stage]["research"]["costs"][resource]
                    for row in rows) for resource in RESOURCES}}
                for stage, name in enumerate(STAGES)]}
    return {"protocol": plan, "completed_cases": len(cases), "totals": totals,
        "paired_cost_differences": [{"seed": case["seed"], **case["paired_effect"]} for case in cases],
        "external_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(8601, 8609)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = protocol(args.seeds)
    (args.output / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_case, seed, plan): seed for seed in args.seeds}
        for future in as_completed(pending):
            seed = pending[future]
            try:
                case = future.result()
            except Exception as error:
                (args.output / f"failure_{seed}.json").write_text(json.dumps({
                    "seed": seed, "error_type": type(error).__name__, "error": str(error)}, indent=2) + "\n")
                raise
            cases.append(case)
            (args.output / f"case_{seed}.json").write_text(json.dumps(case, indent=2) + "\n")
            print(json.dumps({"seed": seed, "solved": {arm: case["results"][arm]["solved"]
                for arm in ARMS}, "paired_effect": case["paired_effect"]}), flush=True)
    cases.sort(key=lambda case: case["seed"])
    summary = summarize(cases, plan)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed_cases": len(cases), "totals": summary["totals"]}), flush=True)


if __name__ == "__main__":
    main()
