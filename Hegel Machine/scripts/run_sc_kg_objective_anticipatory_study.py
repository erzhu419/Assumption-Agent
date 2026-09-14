#!/usr/bin/env python3
"""Evaluate prospective rebuilt-head forecasts using already purchased vectors."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import run_sc_kg_objective_incumbent_study as strict


study, validation, passive = strict.study, strict.validation, strict.passive
BASELINE, STRICT, METHOD = strict.BASELINE, strict.METHOD, "objective_anticipatory_v049"
FRESH_SEEDS = (1843011749, 1843111771, 1843211789)
DEVELOPMENT = (*strict.DEVELOPMENT,
    "objective_incumbent_3_outlier_then_change_noise_0.03",
    "terminal_value_3_mechanism_change_noise_0.03",
    "terminal_value_3_outlier_then_change_noise_0.03")


def references(source):
    original = study.read(source / "protocol.json")
    _, baseline, _ = strict.references(Path(original["source_validation_study"]))
    rows, refs = {BASELINE: baseline, STRICT: {}}, {BASELINE: {}, STRICT: {}}
    for ref in study.read(source / "summary.json")["result_references"]:
        key, arm = ref["case_id"], ref["arm"]
        row = study.read(ref["case_record"]) if ref["run_origin"] == "actual_v049_study" else baseline[key]
        rows[arm][key], refs[arm][key] = row, ref
    return original, rows, refs


def entry_indices(row):
    state, spans = row.get("objective_research") or {}, []
    for attempt in state.get("scope_attempts", []):
        if attempt.get("candidate"):
            spans.append((attempt["candidate"]["training_end"] + 1,
                max((sample["global_index"] + 1 for sample in attempt["validation"]), default=64)))
    current = state.get("scope_research") or {}
    if current.get("candidate"):
        spans.append((current["candidate"]["training_end"] + 1, 64))
    spans.extend((accepted["model"]["training_end"] + 1, accepted["stage"])
                 for accepted in row.get("objective_scope_revisions", []))
    installed = min((accepted["stage"] for accepted in row.get("objective_scope_revisions", [])), default=65)
    return [sample["context"]["global_index"] for sample in (row.get("recovery") or {}).get("observations", [])
            if sample["context"].get("purpose") == "validation"
            and sample["context"]["global_index"] < installed
            and any(start <= sample["context"]["global_index"] < end for start, end in spans)]


def prepare(source, output):
    original, rows, refs = references(source)
    cases = deepcopy(original["cases"])
    affected = [case["case_id"] for case in cases if entry_indices(rows[BASELINE][case["case_id"]])]
    if len(affected) != 146 or set(affected) & set(strict.CONTROLS):
        raise ValueError("the planned146 prospective-comparator entries and six controls differ")
    prior = Path(original["source_validation_study"])
    for case in cases:
        if case["cohort"] == "fresh27":
            case["initial_research_origin"] = str((source / "research" / f"{case['research_id']}.json").resolve())
            case["cohort"] = "historical_v049_27"
        else:
            root = prior if case["cohort"] == "historical_v048_27" else Path(original["historical_measurement_source"])
            case["initial_research_origin"] = str((root / "research" / f"{case['research_id']}.json").resolve())
    templates = [case for case in cases if case["task"]["seed"] == strict.FRESH_SEEDS[0]]
    if len(templates) != 9 or set(FRESH_SEEDS) & {case["task"]["seed"] for case in cases}:
        raise ValueError("the next confirmation requires three unused seeds and nine profiles each")
    fresh = []
    for index, seed in enumerate(FRESH_SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case.pop("initial_research_origin")
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case.update(case_id=f"objective_anticipatory_{index}_{case['profile']}_noise_{noise:.2f}",
                research_id=f"objective_anticipatory_{index}_{problem}_noise_{noise:.2f}", cohort="fresh27")
            fresh.append(case)
    reused = [case["case_id"] for case in cases if case["case_id"] not in set(affected) | set(strict.CONTROLS)]
    cases.extend(fresh)
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation", "frozen_baseline_source")}
    spec.update(experiment="anticipatory_objective_incumbent_comparison", cases=cases,
        arms=[BASELINE, METHOD], reference_arm=STRICT, source_incumbent_study=str(source.resolve()),
        prior_measurement_source=str(prior.resolve()),
        frozen_method_source=str((output / "method_source_anticipatory").resolve()),
        affected_cases=affected, unchanged_controls=list(strict.CONTROLS), reused_cases=reused,
        development_cases=list(DEVELOPMENT), fresh_cases=[case["case_id"] for case in fresh],
        fresh_seeds=list(FRESH_SEEDS), expected_new_optimizer_runs=206,
        expected_new_online_calls=6592, expected_new_initial_calls=384,
        expected_new_evaluation_calls=1920, expected_new_physical_calls=8896,
        method="When a frozen objective candidate coexists with a validating constraint proposal and no "
            "installed objective head, forecast the proposed objective-scope GP before the paid vector. "
            "At actual rebuild, compare with these prospectively recorded opponent errors. Preserve "
            "strict reset when no applicable prospective evidence exists, unchanged empirical support "
            "threshold, and the passive continuation budget. Count extra GP fits/forecasts separately.",
        selection="All146 existing463 paths where a real constraint-validation call occurs during the frozen "
            "objective candidate lifetime and before objective-head installation. This includes26 paths "
            "without actual objective rebuild, so speculative-model compute remains charged. Six unchanged "
            "controls run again;311 other baseline paths are reused. Fifteen development cases precede full work.",
        reuse_rule="The new prospective comparator requires the documented pre-call overlap. Strict reset "
            "also requires a frozen candidate at objective rebuild. All120 existing rebuild entries are included "
            "in the146 prospective entries. Outside them, reuse the existing v0.48 path except six actual controls.",
        accounting="152 old new-method runs plus54 fresh paired baseline/new runs cost6592online. Reuse all "
            "old initialization, evaluation, baseline and strict-reference outcomes. New shared acquisition "
            "costs384initial and1920evaluation:8896physical calls. Development15 costs480online included once. "
            "Fresh27 runs only after development is judged promising. No new strict49, vanilla or CodeAgent runs.",
        limitations="Old463 include the failed strict49 development and confirmation sample, so they are "
            "development evidence for this variant. Report the independently predeclared fresh27 separately. "
            "Logical search compute and extra GP fits/time are different costs; do not combine them as one unit.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved anticipatory-comparison protocol differs")
    if not path.exists():
        study.write(path, spec)
    study.write(output / "entry_inventory.json", {"source_cases": 463, "affected": [
        {"case_id": key, "eligible_pre_call_indices": entry_indices(rows[BASELINE][key]),
         "actual_objective_rebuild": bool(strict.rebuild_entries(rows[BASELINE][key]))} for key in affected],
        "controls": list(strict.CONTROLS), "reused": reused})
    return spec, rows, refs


def measurements(spec, output, cases):
    research = {}
    for case in cases:
        if case["cohort"] == "fresh27":
            continue
        key = case["research_id"]
        if key not in research:
            row = study.read(case["initial_research_origin"])
            if not row.get("optimizer_initial_environment_state") or (
                row["result"]["costs"]["observation_cost"] + row.get("supplemental_initial_calls", 0) != 32):
                raise ValueError(f"missing paid32-row initialization: {key}")
            research[key] = row
        name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
        target = output / "evaluation" / name
        if not target.exists():
            source = next(Path(root) / "evaluation" / name for root in
                (spec["source_incumbent_study"], spec["prior_measurement_source"])
                if (Path(root) / "evaluation" / name).exists())
            if study.read(source)["integer_points"] != study.evaluation_controls(spec, case):
                raise ValueError("shared evaluator controls differ")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return research


def summarize(spec, output, cases, rows, refs, actual, phase):
    comparisons, metrics = {}, {}
    groups = {"overall": cases, "old463": [case for case in cases if case["cohort"] != "fresh27"],
              "fresh27": [case for case in cases if case["cohort"] == "fresh27"]}
    details = []
    for group, selected in groups.items():
        if not selected:
            continue
        comparisons[group], metrics[group] = {}, {}
        for label, arm in (("baseline_to_anticipatory", BASELINE), ("strict_to_anticipatory", STRICT)):
            available = [case for case in selected if case["case_id"] in rows[arm]]
            if not available:
                continue
            comparison = validation.reuse.comparison(available, rows[arm], rows[METHOD])
            comparisons[group][label] = {key: comparison[key] for key in (
                "feasibility_first", "prediction_loss", "paid_trajectories_changed", "matched_complete_kg")}
            if group == "overall" and arm == BASELINE:
                details = comparison["cases"]
        for arm, mapping in rows.items():
            available = [mapping[case["case_id"]] for case in selected if case["case_id"] in mapping]
            if available:
                metric = passive.metrics(available)
                metrics[group][arm] = {key: metric[key] for key in (
                    "cases", "feasible", "mean_objective_gp_mse", "mean_constraint_gp_mse",
                    "objective_head_adoptions", "objective_research_compute")}
                extra = Counter()
                for row in available:
                    extra.update({key: value for key, value in row["costs"].get("objective_diagnostics", {}).items()
                                  if ("comparator" in key or "prospective" in key or "incumbent_comparison" in key)
                                  and isinstance(value, (int, float))})
                metrics[group][arm]["prospective_gp_work"] = dict(extra)
    research = [study.read(path) for path in (output / "research").glob("*.json")]
    initial = sum(row.get("actual_new_research_calls", 0) + row.get("supplemental_initial_calls", 0) for row in research)
    fresh_seeds = set(spec["fresh_seeds"])
    evaluator = sum(row["simulator_calls"] for row in (study.read(path) for path in (output / "evaluation").glob("*.json"))
                    if row["task"]["seed"] - spec["evaluation"]["seed_offset"] in fresh_seeds)
    online = sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual)
    delta = lambda row: row["objective_gp_true_mse"][1] - row["objective_gp_true_mse"][0]
    brief = {"protocol": str((output / "protocol.json").resolve()), "phase": phase, "cases": len(cases),
        "actual_new_optimizer_runs": len(actual), "actual_new_online_calls": online,
        "actual_new_initial_calls": initial, "actual_new_evaluation_calls": evaluator,
        "actual_new_physical_calls": initial + online + evaluator, "comparisons": comparisons,
        "metrics_by_cohort": metrics, "largest_prediction_increases": sorted(details, key=delta, reverse=True)[:8],
        "largest_prediction_decreases": sorted(details, key=delta)[:8], "limitations": spec["limitations"]}
    prefix = "replay_" if phase == "replay" else ""
    study.write(output / f"{prefix}result_brief.json", brief)
    study.write(output / f"{prefix}summary.json", {"protocol": spec, "result_references": refs})
    print(json.dumps({"phase": phase, "cases": len(cases), "new_physical_calls": initial + online + evaluator,
        "comparisons": comparisons["overall"], "metrics": metrics["overall"]}), flush=True)


def run(spec, output, rows, old_refs, phase):
    cases = spec["cases"] if phase == "all" else [next(case for case in spec["cases"]
            if case["case_id"] == key) for key in spec["development_cases"]]
    new_cases = [case for case in cases if case["case_id"] not in spec["reused_cases"]]
    modules = dict(zip((BASELINE, METHOD), validation.arm_modules(spec).values()))
    study.controlled_problem = validation.active.framework.goal.controlled_problem
    module = modules[BASELINE][0]
    study.Program = importlib.import_module(f"{module.__package__}.language").Program
    study.SCKGEnvironment = module.SCKGEnvironment
    research = measurements(spec, output, new_cases)
    fresh = [case for case in cases if case["cohort"] == "fresh27"]
    if fresh:
        research.update(study.acquire_research(spec, output, fresh))
    rows[METHOD], actual, refs = {}, [], []
    for case in cases:
        key = case["case_id"]
        if key in old_refs[STRICT]:
            refs.append(old_refs[STRICT][key])
        for arm in (BASELINE, METHOD):
            if case["cohort"] != "fresh27" and (arm == BASELINE or key in spec["reused_cases"]):
                row, ref = rows[BASELINE][key], {**old_refs[BASELINE][key], "arm": arm}
            else:
                study.ARMS = (arm,)
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
                ref = {"case_id": key, "arm": arm, "case_record": str((output / "runs" / f"{key}_{arm}.json").resolve()),
                       "run_origin": "actual_anticipatory_study"}
                actual.append(row)
            rows[arm][key] = row
            refs.append(ref)
    study.evaluate(spec, output, new_cases, actual, environment_factory=validation.active.framework.goal.evaluation_environment)
    for row in actual:
        study.write(output / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    summarize(spec, output, cases, rows, refs, actual, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    spec, rows, refs = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "affected": 146, "controls": 6,
            "reuse": 311, "development": 15, "fresh": 27, "new_physical_calls": 8896}), flush=True)
    else:
        run(spec, args.output, rows, refs, args.phase)


if __name__ == "__main__":
    main()
