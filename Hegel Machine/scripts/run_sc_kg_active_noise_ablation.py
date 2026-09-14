#!/usr/bin/env python3
"""Ablate active noise nominations while retaining the frozen v0.46 posterior."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import MethodType, SimpleNamespace

import run_sc_kg_active_noise_study as active


study = active.study
ARM = "posterior_only"


def posterior_only_candidates(self, iteration):
    self._noise_offers = set()
    candidates, sources = self._research_candidates(iteration)
    self._sync_noise_context()
    return candidates, sources


def posterior_only_module(module):
    def initialize(*args, **kwargs):
        algorithm = module.initialize_sc_kg_optimizer(*args, **kwargs)
        algorithm._generate_candidates = MethodType(posterior_only_candidates, algorithm)
        return algorithm

    return SimpleNamespace(SCKGEnvironment=module.SCKGEnvironment,
        HegelProgramProblem=module.HegelProgramProblem,
        initialize_sc_kg_optimizer=initialize)


def prepare(source, output, *, transform_spec=None):
    original = study.read(source / "protocol.json")
    by_id = {case["case_id"]: case for case in original["cases"]}
    cases = [deepcopy(by_id[key]) for key in active.REPLAY_CASES]
    spec = deepcopy(original)
    spec.update(experiment="active_noise_posterior_only_ablation", cases=cases,
        arms=[ARM], active_noise_source=str(source.resolve()),
        expected_new_optimizer_runs=len(cases),
        expected_new_online_calls=len(cases) * (original["total_calls_per_arm"] - 32),
        expected_new_initial_calls=0, expected_new_evaluation_calls=0,
        scope="Only the sixteen prespecified v0.46 development replay cases; no fresh cases.",
        affected_development_cases=[case["case_id"] for case in cases],
        comparison_references=str((source / "replay_summary.json").resolve()),
        ablation="Initialize the same frozen v0.46 algorithm, then replace only the instance's "
            "candidate-generation method with _research_candidates followed by _sync_noise_context. "
            "Keep _noise_offers empty. Native and objective offers may still buy repeats. "
            "Keep posterior fits, predictive weights, GP/fantasy noise, certificates and scopes unchanged.",
        reuse_accounting="Sixteen existing development cases; one new posterior-only arm, "
            "32 online calls each. Reuse all initial data, original baseline/full-arm results and "
            "evaluation measurements. No fresh seeds, new initial measurements or new evaluation measurements.")
    if transform_spec is not None:
        transform_spec(spec)
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved ablation protocol differs")
    research = {}
    evaluations = set()
    for case in cases:
        key = case["research_id"]
        if key not in research:
            original_path = source / "research" / f"{key}.json"
            saved = study.read(original_path)
            if (not saved.get("optimizer_initial_environment_state")
                    or saved["result"]["costs"]["observation_cost"]
                    + saved.get("supplemental_initial_calls", 0) != 32):
                raise ValueError("ablation requires the existing complete paid 32-row initialization")
            saved.update(research_origin=str(original_path.resolve()),
                         actual_new_research_calls=0, actual_new_initial_calls=0)
            research[key] = saved
            destination = output / "research" / f"{key}.json"
            if not destination.exists():
                study.write(destination, saved)
        evaluation = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
        if evaluation not in evaluations:
            original_path = source / "evaluation" / evaluation
            # Reading first makes this phase fail before any optimizer can run
            # if a required already paid evaluation is missing.
            study.read(original_path)
            destination = output / "evaluation" / evaluation
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(original_path, destination)
            evaluations.add(evaluation)
    references = study.read(source / "replay_summary.json")["results"]
    expected = {(case["case_id"], arm) for case in cases for arm in active.ARMS}
    if expected - {(row["case_id"], row["arm"]) for row in references}:
        raise ValueError("each development case needs both completed source references")
    study.write(path, spec)
    return spec, research


def comparison(cases, left, right):
    rows = []
    for case in cases:
        old, new = left[case["case_id"]], right[case["case_id"]]
        rows.append({"case_id": case["case_id"], "split": "development",
            "objective": [row["final"]["true_objective"] for row in (old, new)],
            "feasible": [row["final"]["true_feasible"] for row in (old, new)],
            "objective_gp_true_mse": [row["generalization"]["objective_gp_true_objective_means_mse"]
                                      for row in (old, new)],
            "constraint_gp_true_mse": [row["generalization"]["active_gp_true_constraint_means_mse"]
                                       for row in (old, new)],
            "recommendation": [row["final"]["x_recommended"] for row in (old, new)],
            "paid_trajectory_changed": old["simulations"] != new["simulations"]})
    brief = {"cases": rows}
    active.previous.previous.add_outcome_groups(brief)
    comparable = [case["case_id"] for case in cases if all(
        "exact_kg_stages" in mapping[case["case_id"]]["costs"] for mapping in (left, right))]
    return {"feasibility_first": brief["feasibility_first"]["overall"],
            "prediction_loss": brief["prediction_loss_by_split"]["overall"],
            "paid_trajectories_changed": sum(row["paid_trajectory_changed"] for row in rows),
            "matched_complete_kg": {"paired_cases": len(comparable),
                "unavailable_pairs": len(cases) - len(comparable),
                "seconds": [sum(event.get("t_kg_compute") or 0 for key in comparable
                    for event in mapping[key]["costs"]["exact_kg_stages"]) for mapping in (left, right)]},
            "cases": rows}


def summarize(spec, output, results):
    source_rows = study.read(spec["comparison_references"])["results"]
    references = {arm: {row["case_id"]: row for row in source_rows if row["arm"] == arm}
                  for arm in active.ARMS}
    posterior_only = {row["case_id"]: row for row in results}
    by_arm = {**references, ARM: posterior_only}
    cases = spec["cases"]
    brief = {"protocol": str((output / "protocol.json").resolve()),
        "cases": len(cases), "actual_new_optimizer_calls": sum(
            row["costs"]["optimizer_new_simulator_calls"] for row in results),
        "actual_new_initial_calls": 0, "actual_new_evaluation_calls": 0,
        "comparisons": {
            "baseline_to_posterior_only": comparison(cases, references[active.ARMS[0]], posterior_only),
            "posterior_only_to_full": comparison(cases, posterior_only, references[active.ARMS[1]]),
            "baseline_to_full": comparison(cases, references[active.ARMS[0]], references[active.ARMS[1]])},
        "noise_revision": {arm: active.noise_metrics([mapping[case["case_id"]] for case in cases])
                           for arm, mapping in by_arm.items()},
        "interpretation": "Each comparison lists [left, right]; improved means the right arm wins "
            "under the existing feasibility-first comparison. This is a development-set ablation. "
            "Noise/fantasy timing is included in complete KG timing, not added to it."}
    study.write(output / "summary.json", {"protocol": spec, "results": results})
    study.write(output / "result_brief.json", brief)
    print(json.dumps({"cases": len(cases), "new_online_calls": brief["actual_new_optimizer_calls"],
        "comparisons": {name: value["feasibility_first"] for name, value in brief["comparisons"].items()}}),
        flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    spec, research = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "cases": len(spec["cases"]),
            "new_online_calls": spec["expected_new_online_calls"], "new_initial_calls": 0,
            "new_evaluation_calls": 0, "research_groups_reused": len(research),
            "protocol": str((args.output / "protocol.json").resolve())}))
        return
    modules = active.arm_modules(spec)
    module, policy, recovery = modules[active.ARMS[1]]
    study.ARMS = (ARM,)
    study.controlled_problem = active.framework.goal.controlled_problem
    results = study.optimize(spec, args.output, spec["cases"], research,
        arm_modules={ARM: (posterior_only_module(module), policy, recovery)})
    study.evaluate(spec, args.output, spec["cases"], results,
        environment_factory=active.framework.goal.evaluation_environment)
    summarize(spec, args.output, results)


if __name__ == "__main__":
    main()
