#!/usr/bin/env python3
"""Paired transfer of purchased instrument-boundary procedures to new apparatuses."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from math import ceil, floor, isfinite
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.instrument_hypotheses import (
    assess_instrument_hypotheses, instrument_candidates, plan_instrument_disagreement,
)
from hegel_machine.cognition.instrument_experiment_methods import observe_instrument_method
from hegel_machine.cognition.instrument_models import fit_instrument_model, rebind_instrument_model
from hegel_machine.cognition.language import MissingHistory
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab

DEFAULT = ROOT / "artifacts/instrument_boundary_recipe_20261007"
ARMS = ("keep_boundary", "delete_boundary")
COSTS = ("observation_cost", "measurement_cost", "compute_cost", "queries")
CHANNELS = ("fine",)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def settings(seed, objects):
    return {"seed": seed, "apparatus_seed": seed + 1009, "objects": objects,
            "rename_prefix": f"s{seed}_"}


def paired(probe, batch):
    return [(Observation(point, 0., tuple(probe.sequence[:index])), measured)
            for index, (point, measured) in enumerate(zip(probe.sequence, batch.observations))]


def prefix(lab, protocol):
    return lab.random_probe(lab.seed + protocol["prefix_seed_offset"],
                            length=protocol["prefix_rows"], name="paid_prefix")


def fit_prefix(lab, protocol):
    probe = prefix(lab, protocol)
    batch = lab.acquire(probe, CHANNELS)
    budget = SearchBudget(protocol["fit_budget"])
    model = fit_instrument_model(lab.spec, lab.spec_for(CHANNELS), paired(probe, batch), budget)
    return model, {"probe": asdict(probe), "readings": [row.features for row in batch.observations],
                   "costs": {**lab.costs, "compute_cost": budget.spent}}


def boundary(method):
    return method.get("instrument_recipe", {}).get("sequence_mode") == "learned_boundary"


def research(lab, model, book, protocol, task_id, fit_cost, inherited_ids=()):
    steps, compute = [], fit_cost
    for serial in range(protocol["max_rounds"]):
        planning = SearchBudget(protocol["planning_budget_per_round"])
        plan = plan_instrument_disagreement(model, lab.spec, lab.spec_for(CHANNELS), planning,
            max_length=protocol["max_probe_length"], serial=serial,
            tolerance=protocol["tolerance"], methods=book)
        compute += planning.spent
        step = {key: deepcopy(plan[key]) for key in (
            "reason", "sequence_mode", "target_strategy", "field_names", "method_id",
            "parent_method_id", "method_attempts", "predicted_separation") if key in plan}
        step.update(round=serial, planning_cost=planning.spent,
                    inherited_method=plan.get("method_id") in inherited_ids)
        steps.append(step)
        if plan["probe"] is None:
            break
        batch = lab.acquire(plan["probe"], CHANNELS)
        scoring = SearchBudget(protocol["assessment_budget_per_round"])
        assessment = assess_instrument_hypotheses(model, lab.spec, lab.spec_for(CHANNELS),
            paired(plan["probe"], batch), scoring, protocol["tolerance"])
        compute += scoring.spent
        feedback = observe_instrument_method(book, plan, assessment, model, lab.spec,
            task_id=task_id, observation_rows=len(batch.observations),
            measurement_cost=batch.measurement_cost, tolerance=protocol["tolerance"])
        step.update(probe=asdict(plan["probe"]), predictions=plan["predictions"],
            readings=[{name: row.features[name] for name in plan["field_names"]}
                      for row in batch.observations], assessment=assessment, feedback=feedback,
            observation_cost=batch.observation_cost, measurement_cost=batch.measurement_cost,
            actually_distinguished=bool(feedback and (feedback.get("outcome") == "distinguished"
                or feedback["action"] == "learn_instrument_experiment_method")))
    return {"model": model, "book": book.to_dict(), "steps": steps,
            "costs": {**lab.costs, "compute_cost": compute}}


def learn(seed, protocol):
    source_settings = settings(seed, 4)
    with UnifiedObjectLab(**source_settings) as lab:
        model, initial = fit_prefix(lab, protocol)
        source = research(lab, model, ProgramExperimentMethods(), protocol,
                          f"source:{seed}", initial["costs"]["compute_cost"])
    source.update(settings=source_settings, prefix=initial)
    frozen_book = json.loads(json.dumps(source["book"]))
    result = {"source_seed": seed, "source": source, "targets": {}}
    for target in protocol["targets"]:
        target_settings = settings(seed + target["seed_offset"], target["objects"])
        with UnifiedObjectLab(**target_settings) as lab:
            common_model, common_prefix = fit_prefix(lab, protocol)
        record = {"settings": target_settings, "prefix": common_prefix,
                  "initial_model": common_model, "arms": {}}
        for arm in ARMS:
            records = [method for method in frozen_book
                       if arm != "delete_boundary" or not boundary(method)]
            book = ProgramExperimentMethods(json.loads(json.dumps(records)))
            inherited = {method["method_id"] for method in records}
            with UnifiedObjectLab(**target_settings) as lab:
                # The same paid prefix is charged once in each counterfactual.
                lab.acquire(prefix(lab, protocol), CHANNELS)
                record["arms"][arm] = research(lab, deepcopy(common_model), book, protocol,
                    f"target:{target['name']}:{seed}", common_prefix["costs"]["compute_cost"], inherited)
        result["targets"][target["name"]] = record
    return result


def terminal_probes(lab, protocol):
    yield lab.random_probe(lab.seed + protocol["terminal_seed_offset"],
        length=protocol["terminal_random_rows"], name="terminal_random")
    bounds = lab.spec.feature_bounds
    center = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
    for name, (low, high) in bounds.items():
        yield Probe(f"terminal_axis:{name}", tuple({**center, name: value}
                    for value in (low, (low + high) / 2, high)))
    schema = json.loads(lab.spec_for(CHANNELS).context["object_schema"])
    focus = schema["focus_control"]
    low, high = bounds[focus]
    yield Probe("terminal_focus_half_grid", tuple({**center, focus: value / 2}
                for value in range(ceil(2 * low), floor(2 * high) + 1)))


def evaluate(task_settings, model, protocol):
    """Score all frozen survivors; no terminal observation revises a candidate."""
    with UnifiedObjectLab(**task_settings) as lab:
        spec = lab.spec_for(CHANNELS)
        choices = instrument_candidates(model, spec)
        for name, program in rebind_instrument_model(model, spec).items():
            choices.setdefault(name, {"unique": program})
        errors = {name: {key: 0. for key in programs} for name, programs in choices.items()}
        invalid, compute, probes = set(), 0, []
        for probe in terminal_probes(lab, protocol):
            batch = lab.acquire(probe, CHANNELS)
            probes.append({"probe_id": probe.probe_id, "rows": len(batch.observations)})
            for command, measured in paired(probe, batch):
                for name, programs in choices.items():
                    for key, program in programs.items():
                        compute += 1
                        try:
                            error = (program.predict(command) - measured.features[name]) ** 2
                        except (KeyError, MissingHistory, OverflowError):
                            error = float("inf")
                        if not isfinite(error):
                            invalid.add(name)
                        else:
                            errors[name][key] = max(errors[name][key], error)
        missing = sorted(set(spec.feature_names) - set(choices))
        maximum = max((error for values in errors.values() for error in values.values()), default=0.)
        return {"all_fields_terminal_pass": not (missing or invalid) and maximum <= protocol["tolerance"],
            "max_squared_error": None if missing or invalid else maximum,
            "candidate_max_squared_errors": errors, "missing_fields": missing,
            "invalid_prediction_fields": sorted(invalid), "field_count": len(spec.feature_names),
            "candidate_counts": {name: len(programs) for name, programs in choices.items()},
            "probes": probes, "costs": {**lab.costs, "compute_cost": compute}}


def costs_sum(records):
    return {key: sum(record["costs"][key] for record in records) for key in COSTS}


def summarize(results, protocol):
    summary = {"source_count": len(results), "target_count": len(results) * len(protocol["targets"]),
        "sources_with_supported_boundary": sum(any(boundary(method) and method["status"] == "supported"
            for method in row["source"]["book"]) for row in results),
        "source_terminal_pass": sum(row["source"]["terminal"]["all_fields_terminal_pass"] for row in results),
        "source_costs": costs_sum([row["source"] for row in results]),
        "source_terminal_costs": costs_sum([row["source"]["terminal"] for row in results]),
        "arms": {}, "paired_by_source": [],
        "bootstrap": {"unit": "source_seed", "resamples": 10000, "seed": 20261007,
                      "contrast": "keep_boundary minus delete_boundary"}}
    for arm in ARMS:
        targets = [target["arms"][arm] for row in results for target in row["targets"].values()]
        steps = [step for target in targets for step in target["steps"]]
        summary["arms"][arm] = {
            "terminal_pass": sum(target["terminal"]["all_fields_terminal_pass"] for target in targets),
            "inherited_method_uses": sum(step["inherited_method"] and "probe" in step for step in steps),
            "inherited_actual_discriminations": sum(step["inherited_method"] and
                step.get("actually_distinguished", False) for step in steps),
            "actual_discriminations": sum(step.get("actually_distinguished", False) for step in steps),
            "target_only_costs": costs_sum(targets),
            "source_inclusive_costs": costs_sum([*targets, *(row["source"] for row in results)]),
            "terminal_costs": costs_sum([target["terminal"] for target in targets])}
    for row in results:
        by_arm = {arm: costs_sum([target["arms"][arm] for target in row["targets"].values()]) for arm in ARMS}
        passes = {arm: sum(target["arms"][arm]["terminal"]["all_fields_terminal_pass"]
                          for target in row["targets"].values()) for arm in ARMS}
        summary["paired_by_source"].append({"source_seed": row["source_seed"],
            "target_only_costs": by_arm, "terminal_pass": passes,
            "delta": {key: by_arm[ARMS[0]][key] - by_arm[ARMS[1]][key] for key in COSTS},
            "terminal_pass_delta": passes[ARMS[0]] - passes[ARMS[1]]})
    rng, count = Random(20261007), len(results)
    samples = [[rng.randrange(count) for _ in range(count)] for _ in range(10000)]
    intervals = {}
    for key in COSTS:
        deltas = [row["delta"][key] for row in summary["paired_by_source"]]
        means = sorted(sum(deltas[index] for index in sample) / count for sample in samples)
        intervals[key] = {"mean_delta_per_source_two_targets": sum(deltas) / count,
                          "bootstrap_95_ci": [means[249], means[9749]], "total_delta": sum(deltas)}
    summary["paired_cost_differences"] = intervals
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT / "protocol.json")
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--seed", type=int, help="Run one declared source seed instead of the full cohort.")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    seeds = protocol["source_seeds"] if args.seed is None else [args.seed]
    if any(seed not in protocol["source_seeds"] for seed in seeds):
        parser.error("--seed must be declared in the supplied protocol")
    results = []
    # Complete all declared learning before exposing any terminal scores.
    for seed in seeds:
        result = learn(seed, protocol)
        save(args.output / "runs" / f"{seed}.json", result)
        results.append(result)
        print(json.dumps({"source_seed": seed, "phase": "learning_complete"}), flush=True)
    for result in results:
        source = result["source"]
        source["terminal"] = evaluate(source["settings"], source["model"], protocol)
        for target in result["targets"].values():
            for record in target["arms"].values():
                record["terminal"] = evaluate(target["settings"], record["model"], protocol)
        save(args.output / "runs" / f"{result['source_seed']}.json", result)
    summary = summarize(results, protocol)
    save(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "paired_by_source"}, indent=2))


if __name__ == "__main__":
    main()
