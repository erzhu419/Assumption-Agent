#!/usr/bin/env python3
"""Compare independent objective-GP replacement with the actual v0.48 default."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path

import run_sc_kg_objective_anticipatory_study as previous


strict, validation, study, passive = previous.strict, previous.validation, previous.study, previous.passive
BASELINE, METHOD = strict.BASELINE, "objective_replacement_v050"
FRESH_SEEDS = (1943011811, 1943111833, 1943211851)
DEVELOPMENT = previous.DEVELOPMENT


def entry_indices(row):
    installed = min((item["stage"] for item in row.get("objective_scope_revisions", [])), default=65)
    return [sample["context"]["global_index"]
            for sample in (row.get("recovery") or {}).get("observations", [])
            if sample["context"].get("purpose") == "validation"
            and sample["context"]["global_index"] < installed]


def prepare(source, output, variant=None):
    original, old_rows, old_refs = previous.references(source)
    baseline, refs = old_rows[BASELINE], old_refs[BASELINE]
    cases = deepcopy(original["cases"])
    affected = [case["case_id"] for case in cases if entry_indices(baseline[case["case_id"]])]
    controls = [key for key in strict.CONTROLS if key not in affected]
    if (len(cases), len(affected), len(controls)) != (463, 153, 4):
        raise ValueError("the recorded constraint-validation inventory differs")
    prior = Path(original["source_validation_study"])
    for case in cases:
        if case["cohort"] == "fresh27":
            root = source
            case["cohort"] = "historical_v049_27"
        else:
            root = prior if case["cohort"] == "historical_v048_27" else Path(original["historical_measurement_source"])
        case["initial_research_origin"] = str((root / "research" / f"{case['research_id']}.json").resolve())
    templates = [case for case in cases if case["task"]["seed"] == strict.FRESH_SEEDS[0]]
    if len(templates) != 9 or set(FRESH_SEEDS) & ({case["task"]["seed"] for case in cases} | set(previous.FRESH_SEEDS)):
        raise ValueError("confirmation requires three unused seeds and nine existing profiles")
    fresh = []
    for index, seed in enumerate(FRESH_SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case.pop("initial_research_origin")
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case.update(case_id=f"objective_replacement_{index}_{case['profile']}_noise_{noise:.2f}",
                research_id=f"objective_replacement_{index}_{problem}_noise_{noise:.2f}", cohort="fresh27")
            fresh.append(case)
    reused = [case["case_id"] for case in cases if case["case_id"] not in set(affected) | set(controls)]
    cases.extend(fresh)
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    runs = len(affected) + len(controls) + 2 * len(fresh)
    spec.update(experiment="independent_objective_gp_replacement_competition", cases=cases,
        arms=[BASELINE, METHOD], source_incumbent_study=str(source.resolve()),
        prior_measurement_source=str(prior.resolve()),
        frozen_baseline_source=str((output / "baseline_source_v048").resolve()),
        frozen_method_source=str((output / "method_source_v050").resolve()),
        affected_cases=affected, unchanged_controls=controls, reused_cases=reused,
        development_cases=list(DEVELOPMENT), fresh_cases=[case["case_id"] for case in fresh],
        fresh_seeds=list(FRESH_SEEDS), expected_new_optimizer_runs=runs,
        expected_new_online_calls=32 * runs, expected_new_initial_calls=384,
        expected_new_evaluation_calls=1920, expected_new_physical_calls=32 * runs + 2304,
        method="Treat the objective GP implied by a proposed constraint scope as an independent candidate, "
            "using the later of the proposed constraint and active objective scope starts. Compare it with "
            "the retained incumbent on every subsequently purchased vector while the constraint proposal "
            "validates and no objective-specific head is installed. At constraint adoption, replace the "
            "objective GP only with at least four paired labels and the original two-standard-error gain "
            "criterion; a same-call adopted objective candidate also requires direct supporting comparison. "
            "Otherwise retain the incumbent GP and variance, keep the confirmed proposal pending, and "
            "continue comparisons on ordinary paid controls until support or objective-head installation. "
            "At an actual objective GP change, rebase the frozen objective candidate comparison using "
            "prepared direct evidence; historical advantage against the replaced model cannot authorize it.",
        selection="All153 existing paths with real constraint validation before objective-head installation, "
            "regardless of whether an objective candidate is frozen. This includes seven paths beyond the "
            "previous anticipatory-comparison inventory. Four previous no-entry controls are run again. "
            "The same fifteen development cases run before the complete historical and fresh batches.",
        reuse_rule="The replacement method is entered only for a constraint proposal being validated while "
            "no objective-specific head is installed. Reuse306 complete baseline paths without a corresponding "
            "pre-installation validation call; rerun four no-entry controls. Actual prior v0.48 references "
            "include already validated no-entry reuse. Reused paths are not newly executed trajectories.",
        accounting=f"153 affected and4 control runs plus54 fresh paired runs cost{32 * runs} online calls. "
            "Fresh shared acquisition costs384 initial and1920 evaluator calls. Fifteen development runs "
            "cost480 online calls and are included once in the full budget. Each arm has32 initial and32 online "
            "calls. Record extra GP fits, training rows, prediction rows and elapsed fitting time separately. "
            "No new strict49, anticipatory49, vanilla or CodeAgent runs.",
        source_loading="Freeze all nine v0.48 intervention modules plus objective_replacement.py for the "
            "new method. Each arm uses its frozen module package and the unchanged shared support modules.",
        limitations="The existing463 include prior failed-method development and confirmation evidence. "
            "Report the predeclared fresh27 separately and execute them only after development is promising. "
            "Historical timing is not a simultaneous speed comparison. Missing measured objective evaluator "
            "labels remain missing; saved true means are used only for explicitly labelled true MSE.")
    if variant is not None:
        spec.update(variant)
    protocol = output / "protocol.json"
    if protocol.exists() and study.read(protocol) != spec:
        raise ValueError("the saved replacement-competition protocol differs")
    if not protocol.exists():
        study.write(protocol, spec)
    study.write(output / "entry_inventory.json", {"source_cases": len(baseline), "affected": [
        {"case_id": key, "eligible_pre_call_indices": entry_indices(baseline[key])} for key in affected],
        "controls": controls, "reused": reused})
    return spec, baseline, refs


def replacement_behavior(row):
    state = (row.get("objective_research") or {}).get("replacement_research") or {}
    revisions = row.get("scope_revisions", [])
    installs = row["costs"].get("objective_diagnostics", {}).get("replacement_installs", [])
    immediate = sum(any(revision["stage"] == install["stage"]
                        and revision.get("preserved_objective_head") is False
                        for revision in revisions) for install in installs)
    return {"case_id": row["case_id"], "proposal_entry": bool(state.get("events")),
        "proposals": sum(event["action"] == "propose_objective_replacement" for event in state.get("events", [])),
        "constraint_rebuilds": len(revisions),
        "preserved_objective_rebuilds": sum(item.get("preserved_objective_head") is True for item in revisions),
        "raw_objective_rebuilds": sum(item.get("preserved_objective_head") is False for item in revisions),
        "supported_raw_gp_installs": len(installs), "immediate_raw_gp_installs": immediate,
        "later_confirmed_raw_gp_installs": len(installs) - immediate,
        "paired_validation_rows": len(state.get("validation", [])) + sum(
            len(attempt.get("validation", [])) for attempt in state.get("attempts", [])),
        "comparison_rows": state.get("comparison_rows", 0),
        "pending_confirmed": bool(state.get("confirmed")),
        "replacement_support_basis": dict(Counter(
            item["evidence"].get("support_basis", "original_mean_margin")
            for item in state.get("decisions", []) if item.get("replace"))),
        "finished_reasons": dict(Counter(item["reason"] for item in state.get("attempts", [])))}


def summarize(spec, output, cases, baseline, current, refs, actual, phase):
    comparisons, metrics, changed = {}, {}, []
    groups = {"overall": cases, **{cohort: [case for case in cases if case["cohort"] == cohort]
              for cohort in dict.fromkeys(case["cohort"] for case in cases)}}
    for group, selected in groups.items():
        result = validation.reuse.comparison(selected, baseline, current)
        comparisons[group] = {key: result[key] for key in (
            "feasibility_first", "prediction_loss", "paid_trajectories_changed", "matched_complete_kg")}
        comparisons[group]["common_feasible_objective"] = validation.active.common_feasible_objectives(result["cases"])
        metrics[group] = {}
        for arm, mapping in ((BASELINE, baseline), (METHOD, current)):
            rows = [mapping[case["case_id"]] for case in selected]
            metric = passive.metrics(rows)
            behavior = [replacement_behavior(row) for row in rows]
            metric["replacement_comparison_rows"] = sum(row["comparison_rows"] for row in behavior)
            metric["objective_compute_including_replacement_comparisons"] = (
                metric["objective_research_compute"] + metric["replacement_comparison_rows"])
            metric["replacement_behavior"] = {field: sum(row[field] for row in behavior) for field in (
                "proposal_entry", "proposals", "constraint_rebuilds", "preserved_objective_rebuilds",
                "raw_objective_rebuilds", "supported_raw_gp_installs", "immediate_raw_gp_installs",
                "later_confirmed_raw_gp_installs", "paired_validation_rows", "pending_confirmed")}
            support = Counter()
            for row in behavior:
                support.update(row["replacement_support_basis"])
            metric["replacement_support_basis"] = dict(support)
            work = Counter()
            for row in rows:
                work.update({key: value for key, value in row["costs"].get("objective_diagnostics", {}).items()
                             if any(word in key for word in ("replacement", "comparator", "prospective"))
                             and isinstance(value, (int, float))})
            metric["replacement_gp_work"] = dict(work)
            metrics[group][arm] = metric
        if group == "overall":
            changed = [row for row in result["cases"] if row["paid_trajectory_changed"] or any(
                abs(row[field][1] - row[field][0]) > 1e-12 for field in
                ("objective", "objective_gp_true_mse", "constraint_gp_true_mse"))]
    research = [study.read(path) for path in (output / "research").glob("*.json")]
    initial = sum(row.get("actual_new_research_calls", 0) + row.get("supplemental_initial_calls", 0) for row in research)
    fresh_seeds = set(spec["fresh_seeds"])
    evaluator = sum(row["simulator_calls"] for row in (study.read(path) for path in (output / "evaluation").glob("*.json"))
                    if row["task"]["seed"] - spec["evaluation"]["seed_offset"] in fresh_seeds)
    online = sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual)
    brief = {"protocol": str((output / "protocol.json").resolve()), "phase": phase, "cases": len(cases),
        "actual_new_optimizer_runs": len(actual), "actual_new_online_calls": online,
        "actual_new_initial_calls": initial, "actual_new_evaluation_calls": evaluator,
        "actual_new_physical_calls": initial + online + evaluator, "comparisons": comparisons,
        "metrics_by_cohort": metrics, "changed_cases": changed,
        "replacement_cases": [replacement_behavior(current[case["case_id"]]) for case in cases],
        "limitations": spec["limitations"]}
    prefix = "replay_" if phase == "replay" else ""
    study.write(output / f"{prefix}result_brief.json", brief)
    study.write(output / f"{prefix}summary.json", {"protocol": spec, "result_references": refs})
    print(json.dumps({"phase": phase, "cases": len(cases), "new_physical_calls": initial + online + evaluator,
        "comparison": comparisons["overall"]["feasibility_first"],
        "prediction_loss": comparisons["overall"]["prediction_loss"],
        "extra_gp_work": metrics["overall"][METHOD]["replacement_gp_work"]}), flush=True)


def run(spec, output, baseline, old_refs, phase):
    cases = spec["cases"] if phase == "all" else [next(case for case in spec["cases"]
            if case["case_id"] == key) for key in spec["development_cases"]]
    new_cases = [case for case in cases if case["case_id"] not in spec["reused_cases"]]
    modules = dict(zip((BASELINE, METHOD), validation.arm_modules(spec).values()))
    study.controlled_problem = validation.active.framework.goal.controlled_problem
    module = modules[BASELINE][0]
    study.Program = importlib.import_module(f"{module.__package__}.language").Program
    study.SCKGEnvironment = module.SCKGEnvironment
    research = previous.measurements(spec, output, new_cases)
    fresh = [case for case in cases if case["cohort"] == "fresh27"]
    if fresh:
        research.update(study.acquire_research(spec, output, fresh))
    current, actual, refs = {}, [], []
    for case in cases:
        key = case["case_id"]
        for arm in (BASELINE, METHOD):
            if case["cohort"] != "fresh27" and (arm == BASELINE or key in spec["reused_cases"]):
                row = baseline[key]
                ref = {**old_refs[key], "arm": arm, "run_origin": "reused_actual_v048_reference"}
            else:
                study.ARMS = (arm,)
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
                ref = {"case_id": key, "arm": arm, "case_record": str((output / "runs" / f"{key}_{arm}.json").resolve()),
                       "run_origin": "actual_v050_study"}
                actual.append(row)
            (baseline if arm == BASELINE else current)[key] = row
            refs.append(ref)
    study.evaluate(spec, output, new_cases, actual, environment_factory=validation.active.framework.goal.evaluation_environment)
    for row in actual:
        study.write(output / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    summarize(spec, output, cases, baseline, current, refs, actual, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    spec, baseline, refs = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "affected": len(spec["affected_cases"]),
            "controls": len(spec["unchanged_controls"]), "reuse": len(spec["reused_cases"]),
            "development": len(DEVELOPMENT), "fresh": 27,
            "new_physical_calls": spec["expected_new_physical_calls"]}), flush=True)
    else:
        run(spec, args.output, baseline, refs, args.phase)


if __name__ == "__main__":
    main()
