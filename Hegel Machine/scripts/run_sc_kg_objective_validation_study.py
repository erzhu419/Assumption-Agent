#!/usr/bin/env python3
"""Continue frozen objective validation against the actually executed v0.47."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
from statistics import mean
import sys
import types

import run_sc_kg_objective_continuation_replay as continuation


active, reuse, study = continuation.active, continuation.reuse, continuation.study
ARMS = ("objective_continuation_v047", "objective_validation_v048")
FRESH_SEEDS = (1643011651, 1643111669, 1643211683)


def objective_scope_entered(row):
    state = row.get("objective_research") or {}
    return bool(state.get("events") or state.get("scope_research")
        or state.get("scope_attempts") or state.get("scope_revisions")
        or row.get("objective_scope_revisions")
        or any(item.get("reason") == "validate_objective_scope"
               for item in state.get("requests", []))
        or any(item["request"].get("reason") == "validate_objective_scope"
               for item in state.get("shared_outcomes", [])))


def source_results(source):
    old = {row["case_id"]: row for row in study.read(source / "summary.json")["results"]
           if row["arm"] == active.ARMS[0]}
    actual = {row["case_id"]: row for row in study.read(
        source / "objective_continuation/replay/summary.json")["results"]}
    return old, actual


def prepare(source, output):
    old, actual = source_results(source)
    spec = deepcopy(study.read(source / "protocol.json"))
    for field in ("source_change", "old_recorded_summary", "development_source", "first_controller_source",
                  "baseline_method", "baseline_scope", "historical_baseline_arm", "historical_baseline_origin",
                  "affected_development_cases", "actual_control_cases", "actual_unchanged_control",
                  "reused_control_cases", "new_method"):
        spec.pop(field, None)
    cases = spec["cases"]
    for case in cases:
        case["cohort"] = "historical382" if case["split"] == "development" else "historical27"
    affected = [case["case_id"] for case in cases if objective_scope_entered(old[case["case_id"]])]
    untouched = [case["case_id"] for case in cases if case["case_id"] not in affected]
    controls = [key for key in active.REPLAY_CASES if key in untouched]
    missing_baseline = [key for key in affected if key not in actual]
    old_seed = next(case["task"]["seed"] for case in cases if case["split"] == "confirmation")
    templates = [case for case in cases if case["task"]["seed"] == old_seed]
    if len(templates) != 9 or set(FRESH_SEEDS) & {case["task"]["seed"] for case in cases}:
        raise ValueError("fresh confirmation requires nine profiles and three unused seeds")
    fresh = []
    for index, seed in enumerate(FRESH_SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case.update(case_id=f"objective_validation_{index}_{case['profile']}_noise_{noise:.2f}",
                research_id=f"objective_validation_{index}_{problem}_noise_{noise:.2f}", cohort="fresh27")
            fresh.append(case)
    cases.extend(fresh)
    spec.update(experiment="budgeted_continuation_of_frozen_objective_candidates",
        arms=list(ARMS), source_study=str(source.resolve()),
        frozen_baseline_source=str((output / "baseline_source_v047").resolve()),
        frozen_method_source=str((output / "method_source_v048").resolve()),
        baseline_scope="v0.47 contains the executed v0.42 policy plus objective validation resumption "
            "during unresolved constraint research and ordinary paid-label sharing. Sixteen actual "
            "v0.47 outcomes exist; other affected cases require new baseline executions.",
        new_method="Retain a frozen objective candidate when its first four predictions have positive "
            "but uncertain improvement. Continue in affordable pairs chosen for discrimination and "
            "terminal value, without weakening the existing lower-gain adoption threshold. Retain "
            "shared paid controls and reject non-positive candidates. No active-noise changes.",
        baseline_actual_source=str((source / "objective_continuation/replay/summary.json").resolve()),
        replay_cases=list(active.REPLAY_CASES), affected_cases=affected,
        unchanged_entry_reuse=untouched, actual_unchanged_controls=controls,
        missing_actual_baseline_cases=missing_baseline,
        fresh_seeds=list(FRESH_SEEDS), fresh_cases=[case["case_id"] for case in fresh],
        expected_new_optimizer_runs=len(missing_baseline) + len(affected) + len(controls) + 2 * len(fresh),
        expected_new_online_calls=32 * (len(missing_baseline) + len(affected) + len(controls) + 2 * len(fresh)),
        expected_new_initial_calls=384, expected_new_evaluation_calls=1920,
        scope="409 historical cases plus27 fresh confirmation/stress cases: three seeds and nine profiles "
            "per seed fixed before acquisition. First run the original sixteen development cases. "
            "Freeze the method before acquiring any of the new measurements; preserve all regressions.",
        reuse_rule="Reuse an old v0.42 trajectory only when its complete objective state contains no "
            "scope event, active scope, attempt, revision or validation request/outcome. Both v0.47 "
            "and v0.48 changes require entry into objective scope validation. Run six such controls "
            "from the sixteen-case replay; reuse the other unchanged method paths. The remaining "
            "179 cases require actual v0.47 baselines; ten exist, so acquire the missing169.",
        reuse_accounting="All historical initialization and evaluator measurements already exist. The first sixteen "
            "v0.48 runs cost512 online calls. Full historical coverage needs169 additional v0.47 "
            "runs and185 v0.48 runs, including those sixteen once:354 runs/11328 online calls. "
            "Both arms run on27 fresh cases:54 additional runs/1728 online calls. Fresh initialization384 "
            "and evaluation1920 give408 actual optimizer runs/13056 online and15360 physical calls. "
            "Reused trajectories are not new optimizer executions. No vanilla/CodeAgent runs.")
    inventory = {"historical_cases": len(old), "fresh_cases": len(fresh), "affected_cases": affected,
        "no_objective_scope_entry": untouched, "actual_no_entry_controls": controls,
        "actual_v047_available": sorted(actual), "missing_v047_baselines": missing_baseline,
        "affected_by_previous_split": dict(Counter(old[key]["split"] for key in affected)),
        "previous_completed_scope_attempts": [
            {"case_id": key, "status": item.get("status"),
             "validation_count": len(item.get("validation", [])),
             "mean_gain": mean(row["previous_error"] - row["candidate_error"]
                 for row in item["validation"]) if item.get("validation") else None,
             "last_index": item["validation"][-1]["global_index"] if item.get("validation") else None}
            for key, row in old.items()
            for item in (row.get("objective_research") or {}).get("scope_attempts", [])]}
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the prepared objective-validation protocol differs")
    study.write(path, spec)
    study.write(output / "entry_inventory.json", inventory)
    return spec, old, actual


def reuse_measurements(spec, output, cases):
    source = Path(spec["source_study"])
    research = {}
    for case in cases:
        if case["cohort"] == "fresh27":
            continue
        key = case["research_id"]
        if key not in research:
            saved = study.read(source / "research" / f"{key}.json")
            if (not saved.get("optimizer_initial_environment_state")
                    or saved["result"]["costs"]["observation_cost"]
                    + saved.get("supplemental_initial_calls", 0) != 32):
                raise ValueError("requires the existing paid32-row initialization")
            research[key] = saved
        name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
        original = source / "evaluation" / name
        study.read(original)
        destination = output / "evaluation" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(original, destination)
    fresh = [case for case in cases if case["cohort"] == "fresh27"]
    if fresh:
        research.update(study.acquire_research(spec, output, fresh))
    return research


def arm_modules(spec):
    result = {}
    for arm, field in zip(ARMS, ("frozen_baseline_source", "frozen_method_source")):
        directory = Path(spec[field])
        expected = list(Path(spec["frozen_baseline_source"]).glob("*.py"))
        if not expected or any(not (directory / path.name).exists() for path in expected):
            raise ValueError(f"the full source snapshot must be frozen before execution: {directory}")
        name = f"hegel_machine.cognition._objective_validation_{arm}"
        package = types.ModuleType(name)
        package.__path__ = [str(directory), str(study.ROOT / "src/hegel_machine/cognition")]
        sys.modules[name] = package
        modules = {key: importlib.import_module(f"{name}.{key}") for key in
                   ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
        result[arm] = (modules["sc_kg_environment"], modules["prior_adoption"].PriorAdoption,
                       modules["joint_model_revision"].JointModelRevision)
        if arm == ARMS[0]:
            study.MetaPriorResearch = importlib.import_module(f"{name}.meta_prior_research").MetaPriorResearch
    return result


def objective_behavior(row):
    result = continuation.objective_behavior(row)
    state = row.get("objective_research") or {}
    active_scope = state.get("scope_research") or {}
    result.update(final_scope_status=active_scope.get("status"),
        final_validation_target=active_scope.get("validation_target"),
        final_continuation_plan=active_scope.get("continuation_plan"),
        final_validation_rounds=active_scope.get("validation_rounds", []),
        completed_validation_rounds=[attempt.get("validation_rounds", [])
                                     for attempt in state.get("scope_attempts", [])])
    return result


def comparator_rebuild_diagnostic(rows):
    continued = 0
    affected = []
    for row in rows:
        state = row.get("objective_research") or {}
        outcomes = {item["context"]["global_index"]: item for item in
                    [*state.get("outcomes", []), *state.get("shared_outcomes", [])]
                    if item["request"].get("reason") == "validate_objective_scope"}
        for adopted in row.get("objective_scope_revisions", []):
            indices = adopted["validation_indices"]
            if len(indices) <= 4:
                continue
            continued += 1
            for revision in row.get("scope_revisions", []):
                stage = revision["stage"]
                if revision.get("preserved_objective_head") is not False or not min(indices) < stage <= max(indices):
                    continue
                losses = {}
                for label, selected in (("before_rebuild", [i for i in indices if i < stage]),
                                        ("after_rebuild", [i for i in indices if i >= stage])):
                    measured = [outcomes[index] for index in selected]
                    losses[label] = {"count": len(measured), **{
                        name: mean((item["objective"] - item["request"][field]) ** 2 for item in measured)
                        for name, field in (("candidate_mse", "candidate_mean"), ("previous_mse", "previous_mean"))}}
                affected.append({"case_id": row["case_id"], "adoption_stage": adopted["stage"],
                    "comparator_rebuild_stage": stage, "validation_indices": indices, "measured_losses": losses})
    return {"continued_adoptions": continued, "cross_rebuild_adoptions": len(affected),
        "cross_rebuild_cases": len({item["case_id"] for item in affected}), "cases": affected,
        "interpretation": "Read-only explanation of executed adoption: a constraint scope rebuild with "
            "preserved_objective_head=false changes the previous objective GP. Validation indices before "
            "and after that stage therefore compare against different previous predictors. No method changes "
            "or new measurements are made for this diagnostic."}


def summarize(spec, output, cases, results, phase):
    by_arm = {arm: {row["case_id"]: row for row in results if row["arm"] == arm} for arm in ARMS}
    groups = {"overall": cases}
    if phase == "all":
        groups.update({cohort: [case for case in cases if case["cohort"] == cohort]
                       for cohort in ("historical382", "historical27", "fresh27")})
    actual = [row for row in results if row.get("run_origin") == "actual_v048_study"]
    metrics = {}
    for arm, rows in by_arm.items():
        states = [row.get("objective_research") or {} for row in rows.values()]
        attempts = [item for state in states for item in state.get("scope_attempts", [])]
        active_scopes = [state["scope_research"] for state in states if state.get("scope_research")]
        validations = [item for state in states for item in state.get("scope_revisions", [])]
        validation = [item for state in states for item in
            [*state.get("outcomes", []), *state.get("shared_outcomes", [])]
            if item["request"].get("reason") == "validate_objective_scope"]
        metrics[arm] = {"scope_revisions": sum(len(row.get("objective_scope_revisions", [])) for row in rows.values()),
            "validation_observations": len(validation),
            "completed_attempt_statuses": dict(Counter(item.get("status") for item in attempts)),
            "current_scope_statuses": dict(Counter(item.get("status") for item in active_scopes)),
            "continued_research_scopes": sum(len(item.get("validation", item.get("validation_indices", []))) > 4
                                             for item in [*attempts, *active_scopes, *validations]),
            "continuation_requests": sum("continuation_plan" in item["request"] for item in validation),
            "events": dict(Counter(event.get("action") for state in states for event in state.get("events", []))),
            "objective_research_compute": sum(state.get("compute_cost", 0) for state in states),
            "mean_objective_gp_mse": mean(row["generalization"]["objective_gp_true_objective_means_mse"]
                                         for row in rows.values()),
            "mean_constraint_gp_mse": mean(row["generalization"]["active_gp_true_constraint_means_mse"]
                                          for row in rows.values())}
    fresh_research = {case["research_id"] for case in cases if case["cohort"] == "fresh27"}
    fresh_evaluation = {(case["next_problem"], case["task"]["seed"], case["task"]["noise"])
                        for case in cases if case["cohort"] == "fresh27"}
    initial_calls = sum(study.read(output / "research" / f"{key}.json")["actual_new_initial_calls"]
                        for key in fresh_research)
    evaluation_calls = sum(study.read(output / "evaluation" / f"{problem}_{seed}_{noise:.2f}.json")["simulator_calls"]
                           for problem, seed, noise in fresh_evaluation)
    brief = {"protocol": str((output / "protocol.json").resolve()), "phase": phase,
        "cases": len(cases), "actual_new_optimizer_runs": len(actual),
        "actual_new_online_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual),
        "actual_new_initial_calls": initial_calls, "actual_new_evaluation_calls": evaluation_calls,
        "comparisons": {key: reuse.comparison(selected, *[by_arm[arm] for arm in ARMS])
                        for key, selected in groups.items()},
        "objective_behavior": metrics,
        "comparator_rebuild_diagnostic": comparator_rebuild_diagnostic(by_arm[ARMS[1]].values()),
        "behavior": [{"case_id": case["case_id"], **{arm: objective_behavior(
            by_arm[arm][case["case_id"]]) for arm in ARMS}} for case in cases],
        "interpretation": "Paired replay against v0.47; historical and fresh cases remain separate. "
            "Feasibility and objective outcomes, model prediction and validation behavior remain separate. "
            "New validation consumes the same N64 budget. Complete KG timing is not added twice."}
    for group in brief["comparisons"].values():
        paired = group["cases"]
        group["common_feasible_objective"] = active.common_feasible_objectives(paired)
        group["both_infeasible"] = [{"case_id": row["case_id"], "objective": row["objective"]}
                                    for row in paired if row["feasible"] == [False, False]]
        group["feasibility_changes"] = [{"case_id": row["case_id"], "feasible": row["feasible"],
                                        "objective": row["objective"]}
                                       for row in paired if row["feasible"][0] != row["feasible"][1]]
    stem = "replay_" if phase == "replay" else ""
    study.write(output / f"{stem}result_brief.json", brief)
    # Keep full records once in the actual run files; references avoid copying the
    # 200 MB historical summary into every later experiment directory.
    study.write(output / f"{stem}summary.json", {"protocol": spec,
        "result_references": [{"case_id": row["case_id"], "arm": row["arm"],
            "run_origin": row.get("run_origin"), "original_result": row.get("original_result"),
            "case_record": str((output / "runs" / f"{row['case_id']}_{row['arm']}.json").resolve())
                if row.get("run_origin") == "actual_v048_study" else None} for row in results]})
    print(json.dumps({"phase": phase, "cases": len(cases), "new_online_calls": brief["actual_new_online_calls"],
        "comparison": brief["comparisons"]["overall"]["feasibility_first"]}), flush=True)


def run(spec, output, old, existing, phase):
    cases = spec["cases"] if phase == "all" else [
        next(case for case in spec["cases"] if case["case_id"] == key) for key in spec["replay_cases"]]
    modules = arm_modules(spec)
    research = reuse_measurements(spec, output, cases)
    study.controlled_problem = active.framework.goal.controlled_problem
    results = []
    untouched = set(spec["unchanged_entry_reuse"])
    controls = set(spec["actual_unchanged_controls"])
    for case in cases:
        key = case["case_id"]
        for arm in ARMS:
            original = None
            if arm == ARMS[0] and key in existing:
                original, origin = existing[key], "executed_v047_development"
                reference = spec["baseline_actual_source"]
            elif key in untouched and (arm == ARMS[0] or key not in controls):
                original, origin = old[key], "unchanged_objective_scope_entry"
                reference = str(Path(spec["source_study"]) / "summary.json")
            if original is not None:
                row = deepcopy(original)
                row.update(arm=arm, run_origin=origin, original_result=reference)
            else:
                study.ARMS = (arm,)
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
                row["run_origin"] = "actual_v048_study"
            results.append(row)
    study.evaluate(spec, output, cases, results,
        environment_factory=active.framework.goal.evaluation_environment)
    for row in results:
        if row.get("run_origin") == "actual_v048_study":
            study.write(output / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    summarize(spec, output, cases, results, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    spec, old, existing = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "cases": len(spec["cases"]),
            "affected": len(spec["affected_cases"]), "no_entry": len(spec["unchanged_entry_reuse"]),
            "missing_v047_baselines": len(spec["missing_actual_baseline_cases"]),
            "replay_online_calls": 512, "full_online_calls": spec["expected_new_online_calls"]}))
        return
    run(spec, args.output, old, existing, args.phase)


if __name__ == "__main__":
    main()
