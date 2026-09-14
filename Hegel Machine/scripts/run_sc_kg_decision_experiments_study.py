#!/usr/bin/env python3
"""Frozen study of experiment discrimination, objective coverage, and decision value."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import run_sc_kg_active_confirmation_study as accounting

study = accounting.study
SEEDS = (1543010277, 1543110293, 1543210309)
ARMS = ("frozen_v034_active_confirmation", "decision_experiments")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="decision_value_of_distinguishing_experiments",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v034").resolve()),
        historical_baseline_arm="active_confirmation",
        historical_baseline_origin="historical_v034_active_confirmation")
    spec["baseline_method"] = spec.pop("new_method")
    for name in ("joint_model_revision.py", "revision_candidates.py", "sc_kg_environment.py",
                 "prior_adoption.py", "meta_prior_claims.py"):
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError("the actual 0.34 implementation must be saved before decision-experiment edits")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
    for index, seed in enumerate(SEEDS, 1):
        for noise in (0.0, 0.03):
            for profile, problem in (("mechanism_change", "state_policy_rzdt1"),
                                     ("stable_state_policy", "state_policy_rzdt1"),
                                     ("stable_rzdt2", "rzdt2")):
                spec["cases"].append({"case_id": f"decision_experiments_{index}_{profile}_noise_{noise:.2f}",
                    "split": "confirmation", "profile": profile,
                    "task": {"problem": problem, "dimension": 3, "seed": seed, "noise": noise},
                    "next_problem": "rzdt2" if profile == "mechanism_change" else problem,
                    "research_id": f"decision_experiments_{index}_{problem}_noise_{noise:.2f}"})
        spec["cases"].append({"case_id": f"decision_single_outlier_{index}_state_policy_noise_0.03",
            "split": "stress", "profile": "single_observation_outlier",
            "task": {"problem": "state_policy_rzdt1", "dimension": 3, "seed": seed, "noise": 0.03},
            "next_problem": "state_policy_rzdt1",
            "research_id": f"decision_experiments_{index}_state_policy_rzdt1_noise_0.03",
            "observation_outlier": {"global_stage": 45, "online_call": 13, "constraint_delta": 1.0}})
    spec["baseline_scope"] = (
        "The 67 development baselines are actual completed 0.34 active-confirmation runs. "
        "Fresh baseline runs load the saved 0.34 controller, experiment-candidate module, "
        "SC adapter, adoption policy, and claim compiler. No new objective-aware selection "
        "is retroactively inserted into this baseline.")
    spec["new_method"] = {"status": "frozen_before_optimizer_execution",
        "unchanged": "The 0.34 detector, actual anchor confirmation, constraint candidate compilation and competition, and training/future acceptance thresholds remain unchanged.",
        "objective_model": "Fit normalized affine and coordinate-additive quadratic objective programs using the current episode's paid full-vector observations. Select by leave-one-out prediction loss; retain frozen fold models. Prediction variance equals CV loss plus jackknife variance. This Gaussian expected-improvement proxy ranks actions and is not assumed to be a calibrated net-value estimator.",
        "incumbent": "The minimum paid objective in the current scope whose measured constraint satisfies the public constraint upper bound. No evaluator truth enters the controller. Without a feasible measured incumbent, each constraint candidate can offer its minimum predicted feasible objective point.",
        "pool": "Keep the existing public random and boundary pool. Add the analytic box-constrained minimum of the learned separable objective model and 33-point sweeps along every public coordinate through that learned minimum; no task-specific solution is supplied.",
        "selection": "Of four future validation controls, at most two use predicted feasibility and objective opportunity. Each constraint candidate offers its lowest predicted feasible objective point; choose the first offer by Gaussian expected improvement. Choose the second from the predicted-feasible pool by expected improvement times normalized squared distance to the first. Slots without positive opportunity return to disagreement selection. At least two controls retain disagreement times diversity. Missing objective data retains the original selector.",
        "accounting": "Charge all objective fits, held-out fit predictions, fold/pool objective predictions, and constraint predictions to the same existing compute budget. All selected controls consume the original N64 actual observations.",
        "reference_multiplier": 9, "total_calls_per_arm": 64, "initial_calls_per_arm": 32}
    spec["scope"] = (
        "All previous 67 cases are development, including the previous three one-off-outlier "
        "controls. Three previously unused seeds at noise0 and noise0.03 with mechanism change, "
        "stable StatePolicy, and stable RZDT2 make 18 new confirmation cases. Three separate "
        "noisy StatePolicy stress cases add exactly one +1.0 constraint observation at stage45 "
        "without a mechanism change. Compare all groups separately, preserve regression "
        "cases, and keep every optimizer arm within 32 initial plus32 online actual calls.")
    spec["research_initialization"] = (
        "Fresh initial data use MetaPriorResearch with the frozen0.34 claim compiler. "
        "The resulting candidate and actual full-vector data are identical across both arms. "
        "The existing public stratified completion rule supplies 32 initial rows when "
        "research finishes early; historical vectors are reused unchanged.")
    spec["reuse_accounting"] = (
        "Reuse all previously paid initial vectors, actual 0.34 baseline runs, and independent "
        "evaluation measurements. Only the new arm reruns 67 development cases. Both arms "
        "run the 18 fresh confirmation and three fresh outlier stress cases. New and historical "
        "initialization, online observations, compute, and independent evaluation are separate.")
    spec["evaluation"]["additional_requested_metrics"] = (
        "Record which experimental controls provide useful measured objective values, "
        "objective prediction errors on actual future observations, final objective-GP "
        "predictions on the same frozen independent controls when available, and the "
        "best measured point after the mechanism change. Preserve the original constraint "
        "Program/GP errors, confirmation and revision sequence, feasibility, and paid costs.")
    return spec


def arm_modules(spec):
    package_name = "hegel_machine.cognition._decision_experiments_baseline_v034"
    package = types.ModuleType(package_name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[package_name] = package
    baseline = {name: importlib.import_module(f"{package_name}.{name}") for name in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    from hegel_machine.cognition import sc_kg_environment
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    return {ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption,
                      baseline["joint_model_revision"].JointModelRevision),
            ARMS[1]: (sc_kg_environment, PriorAdoption, JointModelRevision)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    args = parser.parse_args()
    study.ARMS = accounting.ARMS = ARMS
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen decision-experiment roster differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 67,
            "confirmation_cases": 18, "single_outlier_stress_cases": 3, "new_optimizer_runs": 67 + 21 * 2}))
        return
    cases = spec["cases"]
    if args.phase == "replay":
        cases = [case for case in cases if case["case_id"] == "active_confirmation_3_mechanism_change_noise_0.03"]
    elif args.phase in ("development", "confirmation", "stress"):
        cases = [case for case in cases if case["split"] == args.phase]
    accounting.prepare_reuse(spec, args.output)
    modules = arm_modules(spec)
    study.MetaPriorResearch = importlib.import_module(
        "hegel_machine.cognition._decision_experiments_baseline_v034.meta_prior_research").MetaPriorResearch
    research = study.acquire_research(spec, args.output, cases)
    if args.phase == "research":
        return
    results = study.optimize(spec, args.output, cases, research, arm_modules=modules)
    evaluations = study.evaluate(spec, args.output, cases, results)
    accounting.summarize(spec, args.output, cases, research, results, evaluations, args.phase)


if __name__ == "__main__":
    main()
