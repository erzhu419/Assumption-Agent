#!/usr/bin/env python3
"""Paired mechanism contrasts in a shared sequence of finite response analogs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations_with_replacement
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean
import sys
from time import process_time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.growth_lab import STAGES
from hegel_machine.cognition.growth_paper_lab import FAMILIES, GrowthPaperLab
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.synthesis import _fit_linear

RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")
ARMS = ("shared", "task_local")


def protocol(seeds):
    return {"experiment": "research_growth_paired_mechanisms", "seeds": list(seeds),
        "families": list(FAMILIES), "family_assignment": "seed modulo 3",
        "stages": list(STAGES), "arms": list(ARMS),
        "research_config": asdict(ResearchConfig(observation_budget=512,
            compute_budget=300000, max_rounds=6, minimum_improvement=1e-7)),
        "measurement_budget": 1536,
        "terminal": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "acceptance": {"deterministic_mse": 1e-5, "stochastic_mean_mse": .03,
            "stochastic_variance_ratio": [.25, 4.]},
        "studies": {
            "expression": "Shared source prefix; add .35*y^3; retain versus remove its program knowledge.",
            "scope": "Learned local response versus frozen old and empirically identified unguarded new branch.",
            "units": "Shared source prefix; anonymous roles and new units; retain versus remove programs.",
            "methods": "Shared measured prefix; new roles and reversed instrument order; retain versus clear methods.",
            "observation_uncertainty": "Same prefix with versus without auxiliary instruments; heteroscedastic calibration.",
            "simplification_return": "Simpler response then old nonlinear return; shared versus fully expressive task-local research."},
        "information_contract": "Three finite algebraic analog families, not real-world domains. "
            "The researcher sees no stage/family labels or latent simulator values. Arms have equal "
            "per-task caps and the same expressive search. Branches inherit identical paid states; "
            "deletion changes only the indicated persisted knowledge. Main-arm execution order is "
            "randomized by seed. All independent terminal evaluations follow all research. "
            "Scope and dense-fit comparisons are prediction diagnostics on the same purchased data, "
            "not additional end-to-end agents. Failures remain in the cohort.",
        "uncertainty": {"unit": "apparatus seed", "method": "family-stratified paired percentile bootstrap",
            "replicates": 10000, "confidence": .95, "seed": 2026100601},
        "external_model_calls": 0}


def delete_programs(state):
    result = deepcopy(state)
    library = KnowledgeLibrary.from_dict(result["library"])
    result["library"] = library.without({item.concept_id for item in library.concepts}).to_dict()
    result["abstractions"] = []
    return result


def delete_methods(state):
    result = deepcopy(state)
    result["methods"]["methods"] = []
    if "research_workflows" in result["library"]:
        result["library"]["research_workflows"]["methods"] = []
    return result


def _train(seed, stage, plan, *, state=None, new_expression=False, controls_only=False, task_seed=None):
    from hegel_machine.cognition.research_growth import GrowthResearcher

    start = process_time()
    config = replace(ResearchConfig(**plan["research_config"]),
                     seed=seed + 10007 * stage if task_seed is None else task_seed)
    researcher = GrowthResearcher(config=config, state=deepcopy(state),
        measurement_budget=plan["measurement_budget"])
    laboratory = GrowthPaperLab(seed=seed, stage=stage, new_expression=new_expression,
                               controls_only=controls_only)
    report = researcher.run(laboratory)
    saved = json.loads(json.dumps(researcher.to_dict()))
    return ({"stage": stage, "stage_name": STAGES[stage], "research": report,
             "environment": deepcopy(laboratory._settings),
             "training_cpu_seconds": process_time() - start}, saved,
            laboratory.paid_observations)


def _prediction(program, data):
    values, steps, calls = [], 0, 0
    for row in data["observations"]:
        context = EvaluationContext(row)
        values.append(context.evaluate(program))
        steps += context.primitive_steps
        calls += context.call_evaluations
    return {"mse": mean((value - row.target) ** 2
            for value, row in zip(values, data["observations"])),
        "mean_mse": mean((value - row.target) ** 2
            for value, row in zip(values, data["mean_observations"])),
        "prediction_rows": len(values), "evaluation_primitive_steps": steps,
        "evaluation_call_executions": calls, "program_size": program.size}, values


def _calibration(values, data, variances):
    positive = [max(value, 1e-12) for value in variances]
    residuals = [(value - row.target) ** 2 for value, row in zip(values, data["observations"])]
    return {"normal_95_coverage": mean(error <= 1.96 ** 2 * var
                for error, var in zip(residuals, positive)),
        "mean_standardized_squared_error": mean(error / var for error, var in zip(residuals, positive)),
        "mean_predicted_variance": mean(positive),
        "variance_mean_absolute_error": mean(abs(value - truth)
                for value, truth in zip(positive, data["noise_variances"]))}


def _evaluate(task, plan, seed, *, program=None):
    report = task["research"]
    laboratory = GrowthPaperLab(**task["environment"])
    data = laboratory.evaluation_rows(report["selected_channels"],
        seed=seed + plan["terminal"]["seed_offset"] + 10007 * task["stage"],
        length=plan["terminal"]["length"], trajectories=plan["terminal"]["trajectories"])
    fitted = Program.from_dict(report["program"]) if program is None else program
    result, predictions = _prediction(fitted, data)
    noise = data["irreducible_noise_variance"]
    uncertainty = report["uncertainty"]
    variances = [uncertainty.get("variance", 0.)] * len(predictions)
    variance_program = uncertainty.get("variance_program")
    if variance_program is not None:
        variance_program = Program.from_dict(variance_program)
        variances = [max(0., variance_program.predict(row)) for row in data["observations"]]
    ratio = mean(variances) / noise if noise else None
    accepted = plan["acceptance"]
    noise_identified = bool(noise and uncertainty["kind"] == "stochastic"
        and accepted["stochastic_variance_ratio"][0] <= ratio <= accepted["stochastic_variance_ratio"][1])
    result.update(solved=bool(result["mean_mse"] <= accepted["stochastic_mean_mse"] and noise_identified
            if noise else result["mse"] <= accepted["deterministic_mse"]),
        confirmed=program is None and report["status"] == "confirmed", uncertainty=uncertainty,
        true_noise_variance=noise, noise_variance_ratio=ratio, noise_identified=noise_identified,
        evaluation_costs={**data["costs"], "compute_cost": len(predictions)})
    if noise:
        result["calibration"] = _calibration(predictions, data, variances)
        result["constant_variance_calibration"] = _calibration(predictions, data,
            [uncertainty.get("variance", 0.)] * len(predictions))
    return result


def _totals(tasks):
    return {"tasks": len(tasks), "solved": sum(task["solved"] for task in tasks),
        "confirmed": sum(task["confirmed"] for task in tasks),
        "mean_mse": mean(task["mean_mse"] for task in tasks),
        "costs": {key: sum(task["research"]["costs"][key] for task in tasks) for key in RESOURCES},
        "training_cpu_seconds": sum(task["training_cpu_seconds"] for task in tasks)}


def _bill(tasks):
    return {"costs": {key: sum(task["research"]["costs"][key] for task in tasks) for key in RESOURCES},
            "training_cpu_seconds": sum(task["training_cpu_seconds"] for task in tasks)}


def _scope_counterfactual(old, learned, rows):
    """Remove the learned guard only when paid rows identify its old branch."""
    def nodes(program):
        yield program
        for child in program.children:
            yield from nodes(child)
    old = old.expanded()
    best, charged = None, 0
    for node in nodes(learned.expanded()):
        if node.op != "if":
            continue
        for index in (1, 2):
            branch = node.children[index]
            charged += 2 * len(rows)
            error = mean((branch.predict(row) - old.predict(row)) ** 2 for row in rows)
            if best is None or error < best[0]:
                best = error, node.children[3 - index]
    return (best[1] if best is not None and best[0] <= 1e-10 else None), charged


def _dense_fit(rows):
    names = sorted(set.intersection(*(set(row.features) for row in rows)))
    features = [Program("feature", name=name) for name in names]
    terms = []
    for degree in range(1, 4):
        for indices in combinations_with_replacement(range(len(features)), degree):
            term = features[indices[0]]
            for index in indices[1:]:
                term = Program("mul", children=(term, features[index]))
            terms.append(term)
    return _fit_linear(terms, rows), (len(terms) + 2) * len(rows)


def run_case(seed, plan):
    order = list(ARMS)
    Random(seed + 31991).shuffle(order)
    reports, snapshots, paid = {}, {}, {}
    for arm in order:
        state, tasks = None, []
        for stage in range(len(STAGES)):
            task, state, observations = _train(seed, stage, plan,
                state=state if arm == "shared" else None)
            tasks.append(task)
            if arm == "shared":
                snapshots[stage] = state
                paid[stage] = observations
        reports[arm] = tasks
    branches = {}
    branch_specs = [
        ("expression_retained", 0, snapshots[0], True, False, seed + 40009),
        ("expression_removed", 0, delete_programs(snapshots[0]), True, False, seed + 40009),
        ("units_removed", 1, delete_programs(snapshots[0]), False, False, None),
        ("methods_removed", 4, delete_methods(snapshots[3]), False, False, None),
        ("controls_only", 3, snapshots[2], False, True, None),
    ]
    Random(seed + 61799).shuffle(branch_specs)
    for name, stage, state, expression, controls_only, task_seed in branch_specs:
        branches[name], _, _ = _train(seed, stage, plan, state=state,
            new_expression=expression, controls_only=controls_only, task_seed=task_seed)
    # Prediction-only contrasts receive the same paid evidence; no terminal
    # rows are used for branch identification or the independent dense fit.
    old = Program.from_dict(reports["shared"][1]["research"]["program"])
    changed = Program.from_dict(reports["shared"][2]["research"]["program"])
    always_new, scope_charge = _scope_counterfactual(old, changed, paid[2])
    dense, dense_charge = _dense_fit(paid[6])
    # Every researcher and diagnostic fit has now finished; independent scoring starts here.
    for tasks in reports.values():
        for task in tasks:
            task.update(_evaluate(task, plan, seed))
    for task in branches.values():
        task.update(_evaluate(task, plan, seed))
    scope = {"activated": always_new is not None,
        "identification_compute_cost": scope_charge,
        "repaired": {key: reports["shared"][2][key] for key in (
            "mse", "mean_mse", "solved", "evaluation_primitive_steps", "program_size")},
        "frozen_old": _evaluate(reports["shared"][2], plan, seed, program=old),
        "always_new": _evaluate(reports["shared"][2], plan, seed, program=always_new)
            if always_new is not None else None}
    simplification = {"dense_fit_compute_cost": dense_charge,
        "learned": {key: reports["shared"][6][key] for key in (
            "mse", "mean_mse", "solved", "evaluation_primitive_steps", "program_size")},
        "dense_fit": _evaluate(reports["shared"][6], plan, seed, program=dense)}
    def pair(retained, control, prefix):
        return {"retained": _totals([retained]), "control": _totals([control]),
            "common_prefix": _bill(prefix),
            "whole_sequence": {"retained": _totals([*prefix, retained]),
                               "control": _totals([*prefix, control])}}
    contrasts = {
        "expression": pair(branches["expression_retained"], branches["expression_removed"], reports["shared"][:1]),
        "units": pair(reports["shared"][1], branches["units_removed"], reports["shared"][:1]),
        "methods": pair(reports["shared"][4], branches["methods_removed"], reports["shared"][:4]),
        "measurement": pair(reports["shared"][3], branches["controls_only"], reports["shared"][:3]),
        "sequence": {"retained": _totals(reports["shared"]), "control": _totals(reports["task_local"])}}
    final = snapshots[7]
    return {"seed": seed, "family": FAMILIES[seed % 3], "arm_execution_order": order,
        "results": {arm: {**_totals(tasks), "tasks": tasks} for arm, tasks in reports.items()},
        "branches": branches, "contrasts": contrasts, "scope": scope, "simplification": simplification,
        "persistent_knowledge": {"concepts": len(final["library"]["concepts"]),
            "methods": final["methods"], "abstractions": final["abstractions"]},
        "external_model_calls": 0}


def _percentile(values, fraction):
    position = fraction * (len(values) - 1)
    left = int(position)
    return values[left] + (position - left) * (values[min(left + 1, len(values) - 1)] - values[left])


def _bootstrap(differences, samples):
    bootstrap = sorted(sum(differences[index] for index in sample) / len(sample) for sample in samples)
    return {"mean_retained_minus_control_per_seed": mean(differences),
        "paired_95_percent_interval": [_percentile(bootstrap, .025), _percentile(bootstrap, .975)],
        "negative": sum(value < -1e-12 for value in differences),
        "zero": sum(abs(value) <= 1e-12 for value in differences),
        "positive": sum(value > 1e-12 for value in differences)}


def summarize(cases, plan, failures=()):
    result = {"protocol": plan, "planned_cases": len(plan["seeds"]), "completed_cases": len(cases),
        "execution_failures": list(failures), "external_model_calls": 0,
        "totals": {}, "mechanisms": {}, "paired_uncertainty": {}}
    if not cases:
        return result
    for arm in ARMS:
        tasks = [task for case in cases for task in case["results"][arm]["tasks"]]
        result["totals"][arm] = {**_totals(tasks),
            "stages": [{"stage": stage, "stage_name": name,
                **_totals([case["results"][arm]["tasks"][stage] for case in cases])}
                for stage, name in enumerate(STAGES)]}
    rng = Random(plan["uncertainty"]["seed"])
    groups = [[index for index, case in enumerate(cases) if case["family"] == family] for family in FAMILIES]
    samples = [sum((rng.choices(group, k=len(group)) for group in groups if group), [])
               for _ in range(plan["uncertainty"]["replicates"])]
    metrics = ("solved", "confirmed", "mean_mse", *RESOURCES, "training_cpu_seconds")
    for contrast in ("sequence", "expression", "units", "methods", "measurement"):
        rows = [case["contrasts"][contrast] for case in cases]
        result["mechanisms"][contrast] = {arm: {
            "solved": sum(row[arm]["solved"] for row in rows),
            "confirmed": sum(row[arm]["confirmed"] for row in rows),
            "mean_mse": mean(row[arm]["mean_mse"] for row in rows),
            "costs": {key: sum(row[arm]["costs"][key] for row in rows) for key in RESOURCES},
            "training_cpu_seconds": sum(row[arm]["training_cpu_seconds"] for row in rows)}
            for arm in ("retained", "control")}
        intervals = {}
        for metric in metrics:
            def value(item):
                return item["costs"][metric] if metric in RESOURCES else item[metric]
            differences = [value(row["retained"]) - value(row["control"]) for row in rows]
            if all(isfinite(value) for value in differences):
                intervals[metric] = _bootstrap(differences, samples)
        result["paired_uncertainty"][contrast] = intervals
    result["scope"] = {"activated": sum(case["scope"]["activated"] for case in cases),
        "repaired_passed": sum(case["scope"]["repaired"]["solved"] for case in cases),
        "frozen_old_passed": sum(case["scope"]["frozen_old"]["solved"] for case in cases),
        "always_new_passed": sum(case["scope"]["always_new"]["solved"]
            for case in cases if case["scope"]["activated"])}
    result["uncertainty_calibration"] = {arm: {
        key: mean(case["results"][arm]["tasks"][5]["calibration"][key] for case in cases)
        for key in ("normal_95_coverage", "mean_standardized_squared_error",
                    "mean_predicted_variance", "variance_mean_absolute_error")}
        for arm in ARMS}
    result["constant_variance_calibration"] = {key: mean(
        case["results"]["shared"]["tasks"][5]["constant_variance_calibration"][key] for case in cases)
        for key in ("normal_95_coverage", "mean_standardized_squared_error",
                    "mean_predicted_variance", "variance_mean_absolute_error")}
    result["simplification"] = {"learned_passed": sum(case["simplification"]["learned"]["solved"] for case in cases),
        "dense_fit_passed": sum(case["simplification"]["dense_fit"]["solved"] for case in cases),
        "learned_terminal_steps": sum(case["simplification"]["learned"]["evaluation_primitive_steps"] for case in cases),
        "dense_fit_terminal_steps": sum(case["simplification"]["dense_fit"]["evaluation_primitive_steps"] for case in cases)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(9400, 9430)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = protocol(args.seeds)
    (args.output / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    cases, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, seed, plan): seed for seed in args.seeds}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                case = future.result()
            except Exception:
                failure = {"seed": seed, "traceback": traceback.format_exc()}
                failures.append(failure)
                (args.output / f"failure_{seed}.json").write_text(json.dumps(failure, indent=2) + "\n")
                print(json.dumps({"seed": seed, "status": "execution_failed"}), flush=True)
                continue
            cases.append(case)
            (args.output / f"case_{seed}.json").write_text(json.dumps(case, indent=2) + "\n")
            print(json.dumps({"seed": seed, "family": case["family"], "solved": {
                arm: case["results"][arm]["solved"] for arm in ARMS},
                "branches": {name: task["solved"] for name, task in case["branches"].items()}}), flush=True)
    cases.sort(key=lambda case: case["seed"])
    summary = summarize(cases, plan, failures)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed_cases": len(cases), "execution_failures": len(failures),
        "totals": {arm: {key: value for key, value in values.items() if key != "stages"}
                   for arm, values in summary["totals"].items()}}), flush=True)


if __name__ == "__main__":
    main()
