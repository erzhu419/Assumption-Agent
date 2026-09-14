#!/usr/bin/env python3
"""Separate observation repair from mechanism revision under the same N64 budget."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import run_sc_kg_active_confirmation_study as accounting

study = accounting.study
SEEDS = (1643010331, 1643110349, 1643210363)
ARMS = ("frozen_v035_decision_experiments", "observation_repair")
BASELINE_MODULES = ("joint_model_revision.py", "revision_candidates.py", "decision_experiments.py",
                    "sc_kg_environment.py", "prior_adoption.py", "meta_prior_claims.py")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="repair_rejected_observation_without_replacing_valid_knowledge",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v035").resolve()),
        historical_baseline_arm="decision_experiments",
        historical_baseline_origin="historical_v035_decision_experiments")
    spec["baseline_method"] = spec.pop("new_method")
    for name in BASELINE_MODULES:
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError(f"the actual pre-edit 0.35 baseline is required: {name}")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
    for index, seed in enumerate(SEEDS, 1):
        for noise in (0., .03):
            for profile, problem in (("mechanism_change", "state_policy_rzdt1"),
                                     ("stable_state_policy", "state_policy_rzdt1"),
                                     ("stable_rzdt2", "rzdt2")):
                spec["cases"].append({"case_id": f"observation_repair_{index}_{profile}_noise_{noise:.2f}",
                    "split": "confirmation", "profile": profile,
                    "task": {"problem": problem, "dimension": 3, "seed": seed, "noise": noise},
                    "next_problem": "rzdt2" if profile == "mechanism_change" else problem,
                    "research_id": f"observation_repair_{index}_{problem}_noise_{noise:.2f}"})
        common = {"split": "stress",
            "task": {"problem": "state_policy_rzdt1", "dimension": 3, "seed": seed, "noise": .03},
            "research_id": f"observation_repair_{index}_state_policy_rzdt1_noise_0.03"}
        spec["cases"].append({**deepcopy(common),
            "case_id": f"repair_single_outlier_{index}_state_policy_noise_0.03",
            "profile": "single_observation_outlier", "next_problem": "state_policy_rzdt1",
            "observation_outlier": {"global_stage": 45, "online_call": 13, "constraint_delta": 1.0}})
        spec["cases"].append({**deepcopy(common),
            "case_id": f"repair_outlier_then_change_{index}_noise_0.03",
            "profile": "outlier_then_change", "next_problem": "rzdt2",
            "change_after_new_calls": 12,
            "observation_outlier": {"global_stage": 41, "online_call": 9, "constraint_delta": 1.0}})
    spec["baseline_scope"] = (
        "Reuse all 88 actual completed 0.35 optimizer runs, initial observations and independent "
        "evaluation measurements. New baseline cases import the six saved 0.35 source modules, "
        "including objective-aware experiment planning. No observation repair enters this baseline.")
    spec["new_method"] = study.read(output / "method.json")
    spec["scope"] = (
        "All 88 previous cases are development. Three previously unused seeds each provide six "
        "ordinary confirmation cases (two noise levels and three profiles), one single-constraint-"
        "outlier control, and one single outlier followed by a real mechanism change. Stress groups "
        "remain separate. Every optimizer arm uses the same 32 initial plus 32 new actual calls.")
    spec["mixed_control"] = (
        "The StatePolicy simulator supplies one +1.0 constraint observation at global stage41 "
        "(online9), with unchanged objective and future observations. Starting at global stage45 "
        "(online13), both actual objective and constraint mechanisms switch to RZDT2. The runtime "
        "receives only public controls and ordinary simulator outputs; changes and disturbances "
        "are report-only. This asks whether observation repair preserves sensitivity to later drift.")
    spec["research_initialization"] = (
        "Fresh initial data and executable programs use the saved 0.35 claim compiler. The existing "
        "public stratified completion rule supplies 32 actual vectors; both arms share these values. "
        "The new stress cases reuse their ordinary same-task noisy StatePolicy initialization.")
    spec["reuse_accounting"] = (
        "Only the new arm reruns 88 development cases. Both arms run 18 fresh ordinary and six "
        "fresh stress cases (136 new optimizer runs, 4352 online calls). Initial research, its "
        "supplement, independent evaluation, and predictor-repair work are accounted separately.")
    return spec


def controlled_problem(base_class, environment, program, case):
    """A report-only disturbance and independently scheduled true change."""
    class ControlledProblem(base_class):
        def __init__(self):
            super().__init__(environment, program)
            from problems.rzdt import RZDT2
            self.next_problem = RZDT2(d=environment._problem.d, L=100,
                sigma=environment._settings["noise"], heteroscedastic=False)
            self.simulation_log = []

        def simulate(self, point, rng=None):
            import numpy as np
            from problems.single_objective import ScalarizedProblem
            change_after = case.get("change_after_new_calls", study.CHANGE_AFTER)
            changed = case["profile"] in ("mechanism_change", "outlier_then_change") and self.simulator_calls >= change_after
            if changed and self.simulator_calls == change_after:
                self.base = ScalarizedProblem(self.next_problem)
            vector = [float(value) for value in self.base.base.simulate(point, rng)]
            disturbance = case.get("observation_outlier")
            delta = (float(disturbance["constraint_delta"])
                if disturbance and self.simulator_calls + 1 == disturbance["online_call"] else 0.)
            vector[2] += delta
            output = [float(self.base.weights[0] * vector[0] + self.base.weights[1] * vector[1]), vector[2]]
            self.simulator_calls += 1
            self.simulation_log.append({"new_call": self.simulator_calls,
                "mechanism": case["next_problem"] if changed else case["task"]["problem"],
                "point": list(map(int, point)), "vector_outputs": vector, "output": output})
            if delta:
                self.simulation_log[-1]["measurement_constraint_disturbance"] = delta
            return np.asarray(output, dtype=float)

    return ControlledProblem()


def arm_modules(spec):
    package_name = "hegel_machine.cognition._observation_repair_baseline_v035"
    package = types.ModuleType(package_name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[package_name] = package
    baseline = {name: importlib.import_module(f"{package_name}.{name}") for name in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    from hegel_machine.cognition import sc_kg_environment
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    study.MetaPriorResearch = importlib.import_module(f"{package_name}.meta_prior_research").MetaPriorResearch
    return {ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption,
                      baseline["joint_model_revision"].JointModelRevision),
            ARMS[1]: (sc_kg_environment, PriorAdoption, JointModelRevision)}


def enrich_repair_summary(spec, output, phase):
    stem = "" if phase == "all" else f"{phase}_"
    summary = study.read(output / f"{stem}summary.json")
    mixed = {case["case_id"]: case for case in spec["cases"] if case["profile"] == "outlier_then_change"}
    compact = study.read(output / f"{stem}compact_summary.json")
    compact["unchanged_mechanism_research"] = [row for row in compact["unchanged_mechanism_research"]
        if row["case_id"] not in mixed]
    compact["mixed_case_sequences"] = []
    compact["observation_repair_records"] = [{"case_id": row["case_id"], "arm": row["arm"],
        "repairs": row.get("observation_repairs", []),
        "rejected_constraint_indices": row.get("rejected_constraint_indices", []),
        "total_paid_calls": row["costs"]["total_simulator_calls"],
        "scope_revisions": len(row["scope_revisions"])}
        for row in summary["results"] if row.get("observation_repairs")]
    repair_costs = ("predictor_snapshots", "observation_repairs", "repair_gp_updates", "repair_hvd_updates",
                   "filtered_replicate_updates", "filtered_fantasy_updates", "observation_repair_seconds")
    fresh = [row for row in summary["results"]
             if row.get("run_origin") != spec["historical_baseline_origin"]]
    compact["costs"].update({f"actual_new_{key}": sum(row["costs"]["recovery_diagnostics"].get(key, 0)
        for row in fresh) for key in repair_costs})
    detail_costs = ("predictor_snapshot_seconds", "repair_mean_predictions", "repair_epistemic_predictions",
                    "repair_noise_predictions", "filtered_epistemic_predictions")
    current_rows = [row for row in fresh if row["arm"] == ARMS[1]]
    compact["repair_detail_recording"] = {}
    for key in detail_costs:
        recorded = [row for row in current_rows if key in row["costs"]["recovery_diagnostics"]]
        compact["repair_detail_recording"][key] = {
            "sum_in_recorded_runs": sum(row["costs"]["recovery_diagnostics"][key] for row in recorded),
            "recorded_runs": len(recorded), "unavailable_runs": len(current_rows) - len(recorded)}
    for group in compact["groups"]:
        for arm in ARMS:
            rows = [row for row in summary["results"]
                    if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                    == (group["split"], group["profile"], group["noise"], arm)]
            group[arm]["observation_repairs"] = sum(len(row.get("observation_repairs", [])) for row in rows)
            group[arm]["rejected_constraint_records_still_charged"] = sum(len(row.get("rejected_constraint_indices", [])) for row in rows)
    for row in summary["results"]:
        if row["case_id"] not in mixed:
            continue
        case = mixed[row["case_id"]]
        stages = [{key: event[key] for key in ("action", "stage", "confirmed", "scope_start", "supported") if key in event}
            for event in row["recovery"]["events"] if event["action"] in
                ("measure_counterexample_confirmation", "reopen_hypothesis_research", "validate_replacement")]
        compact["mixed_case_sequences"].append({"case_id": row["case_id"], "arm": row["arm"],
            "first_changed_stage": 33 + case["change_after_new_calls"], "events": stages,
            "observation_repairs": [{key: record[key] for key in ("stage", "global_index", "confirmation_index") if key in record}
                for record in row.get("observation_repairs", [])],
            "gp_mse": row["generalization"]["active_gp_true_constraint_means_mse"],
            "objective": row["final"]["true_objective"]})
    for aggregate in summary["aggregate"]:
        if aggregate["profile"] == "outlier_then_change":
            rows = [row for row in summary["results"] if row["profile"] == aggregate["profile"] and row["arm"] == aggregate["arm"]]
            aggregate["reopenings_on_unchanged_mechanism"] = sum(event["action"] == "reopen_hypothesis_research"
                and event["stage"] < 33 + mixed[row["case_id"]]["change_after_new_calls"]
                for row in rows for event in row["recovery"]["events"])
    study.write(output / f"{stem}summary.json", summary)
    study.write(output / f"{stem}compact_summary.json", compact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    parser.add_argument("--case", action="append", help="Run selected cases from the frozen roster.")
    args = parser.parse_args()
    study.ARMS = accounting.ARMS = ARMS
    study.controlled_problem = controlled_problem
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen observation-repair method or roster differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 88,
            "confirmation_cases": 18, "stress_cases": 6, "new_optimizer_runs": 136}))
        return
    cases = spec["cases"]
    if args.phase == "replay" and not args.case:
        cases = [case for case in cases if case["case_id"] == "decision_single_outlier_2_state_policy_noise_0.03"]
    elif args.phase in ("development", "confirmation", "stress"):
        cases = [case for case in cases if case["split"] == args.phase]
    if args.case:
        cases = [case for case in cases if case["case_id"] in args.case]
        if not cases or set(args.case) - {case["case_id"] for case in cases}:
            raise ValueError("case selection must name frozen cases in the requested phase")
    accounting.prepare_reuse(spec, args.output)
    modules = arm_modules(spec)
    research = study.acquire_research(spec, args.output, cases)
    if args.phase == "research":
        return
    results = study.optimize(spec, args.output, cases, research, arm_modules=modules)
    evaluations = study.evaluate(spec, args.output, cases, results)
    accounting.summarize(spec, args.output, cases, research, results, evaluations, args.phase)
    enrich_repair_summary(spec, args.output, args.phase)


if __name__ == "__main__":
    main()
