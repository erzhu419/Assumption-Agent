#!/usr/bin/env python3
"""Compare historical full-structure SC-KG with frozen Hegel on paid prefixes."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_sc_kg_goal_completion_study as goal
import run_sc_kg_joint_revision_study as study
import run_sc_kg_unknown_revision_study as unknown
from hegel_machine.cognition.sc_kg_structural_prior import (
    attach_structural_prior, structural_prior_diagnostics,
)

SOURCE = ROOT / "artifacts/sc_kg_goal_completion_20260911"
DEFAULT_OUTPUT = ROOT / "artifacts/sc_kg_structural_prior_comparison_20261007"
KG_COMMIT = "aa1f61e0c7aaf74cc514dbb6774df5a52d4d973a"
NATIVE = "native_quadratic"
STRUCTURAL = "historical_full_structural_prior"
HEGEL = goal.METHOD
ARMS = (NATIVE, STRUCTURAL, HEGEL)


def prepare(output):
    original = study.read(SOURCE / "fresh_confirmation/protocol.json")
    spec = deepcopy(original)
    spec.update(
        experiment="historical_structural_prior_vs_hegel_v053",
        arms=list(ARMS),
        sc_kg_project_root=str((output / "kg_source/SC-OLH-KG").resolve()),
        kg_commit=KG_COMMIT,
        method_source=str(SOURCE / "method_source"),
        source_protocol=str(SOURCE / "fresh_confirmation/protocol.json"),
        comparison="Three systems share 32 paid prefix vectors, 32 further simulator calls, "
        "the historical SC-KG backend, both GP heads, task changes and evaluation controls. "
        "The full-prior arm changes the GP representation using the historical full profile, "
        "preserving its pilot gate and coefficient adaptation. Hegel v0.53 also chooses "
        "research actions and can revise deployed explanations: this is a complete-system "
        "comparison, not an isolated prior-representation or recursion ablation.",
        structural_prior_data="Fit the historical four-property prior on the same current-task "
        "paid prefix. This adapts its acquisition input; it is not the original independent "
        "multi-domain source-training experiment. No new prior-training simulator calls.",
        reused_prefixes=str(SOURCE / "fresh_confirmation/research"),
        reused_evaluation=str(SOURCE / "fresh_confirmation/evaluation"),
        scope="Retrospective development comparison on the nine previously evaluated v0.53 "
        "cases. Rerun all three arms on one extracted historical KG backend; do not count "
        "historical prefix or evaluation measurements as new observations.",
        physical_accounting={"new_initial_calls": 0, "new_evaluation_calls": 0,
                             "new_online_calls_full_roster": 9 * 3 * 32},
    )
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved comparison protocol differs")
    study.write(path, spec)
    return spec


def paid_records(saved):
    state = saved["optimizer_initial_environment_state"]
    return [{"point": point, "output": [.5 * vector[0] + .5 * vector[1], vector[2]],
             "vector_output": vector}
            for acquisition in state["acquisitions"]
            for point, vector in zip(acquisition["integer_points"], acquisition["vector_outputs"])]


def run_fixed_prior(spec, output, case, saved, arm):
    path = output / "runs" / f"{case['case_id']}_{arm}.json"
    if path.exists():
        return study.read(path)
    records = paid_records(saved)
    algorithm, problem = unknown.build_native_optimizer(spec, case, records)
    initial_prior = None
    if arm == STRUCTURAL:
        initial_prior = attach_structural_prior(
            problem, records, seed=case["task"]["seed"] + spec["optimizer_seed_offset"])
        # Construction buys no evidence. Keep the original paid-data loader and
        # its closure over all 32 records, rebuilding only the newly installed GPs.
        algorithm = type(algorithm)(problem, algorithm.config)
    started = perf_counter()
    final = algorithm.run()
    optimizer_seconds = perf_counter() - started
    actual_initial = [[float(value) for value in vector] for _, vector in algorithm.history[:32]]
    if (problem.actual_simulator_calls != 32 or len(algorithm.history) != 64
            or actual_initial != [row["output"] for row in records]
            or [list(point) for point in algorithm.pre_sampling_log["samples"]]
            != [list(row["point"]) for row in records]):
        raise RuntimeError("the fixed prior must use the common 32 rows and exactly 32 new calls")
    controls = study.evaluation_controls(spec, case)
    row = {
        "case_id": case["case_id"], "profile": case["profile"], "arm": arm,
        "task": case["task"], "initial_points": algorithm.pre_sampling_log["samples"],
        "initial_outputs": actual_initial, "optimizer_history_count": len(algorithm.history),
        "simulations": problem.simulation_log,
        "actions": [{key: action.get(key) for key in
                     ("stage", "x_selected", "Y_observed", "mu_before", "action_kind", "n_candidates")}
                    for action in algorithm.iteration_log],
        "costs": {"total_initial_simulator_calls": 32, "optimizer_new_simulator_calls": 32,
                  "total_simulator_calls": 64, "objective_research_compute": 0,
                  "constraint_research_compute": 0, "optimizer_seconds": optimizer_seconds,
                  "native_kg_seconds": sum(action.get("t_kg_compute", 0.) for action in algorithm.iteration_log),
                  "source_fit_seconds": (initial_prior or {}).get("source_fit_seconds", 0.),
                  "initial_basis_fit_seconds": (initial_prior or {}).get("initial_basis_fit_seconds", 0.),
                  "initial_belief_timing": getattr(algorithm, "_last_initial_belief_timing", {})},
        "structural_prior_initial": initial_prior,
        "structural_prior_final": structural_prior_diagnostics(problem) if arm == STRUCTURAL else None,
        "frozen_evaluation_predictions": {"integer_points": controls,
            "objective_gp": [float(algorithm.gpr[0].posterior_mean(point)) for point in controls],
            "active_gp": [float(algorithm.gpr[1].posterior_mean(point)) for point in controls]},
        "final": {key: final.get(key) for key in
                  ("x_recommended", "true_objective", "true_constraint_mean", "true_feasible",
                   "n_simulations", "total_time_sec")},
    }
    study.write(path, row)
    print(json.dumps({"case": case["case_id"], "arm": arm, "final": row["final"]}), flush=True)
    return row


def evaluate_saved(case, row):
    path = SOURCE / "fresh_confirmation/evaluation" / goal.evaluation_name(case)
    evaluation = study.read(path)
    prediction = row["frozen_evaluation_predictions"]
    if prediction["integer_points"] != evaluation["integer_points"]:
        raise ValueError("frozen predictions and historical evaluation controls differ")
    row["generalization"] = {
        "objective_mse": mean((p - y) ** 2 for p, y in
                              zip(prediction["objective_gp"], evaluation["true_objective_means"])),
        "constraint_mse": mean((p - y) ** 2 for p, y in
                               zip(prediction["active_gp"], evaluation["true_constraint_means"])),
        "evaluation_source": str(path), "new_evaluation_calls": 0,
    }
    row["regret"] = float(row["final"]["true_objective"]) - .35
    if row["arm"] in (NATIVE, HEGEL):
        old_arm = unknown.NATIVE if row["arm"] == NATIVE else HEGEL
        old = study.read(SOURCE / "fresh_confirmation/runs" / f"{case['case_id']}_{old_arm}.json")
        row["historical_replay"] = {
            "same_recommendation": old["final"]["x_recommended"] == row["final"]["x_recommended"],
            "objective_difference": row["final"]["true_objective"] - old["final"]["true_objective"],
            "same_feasibility": old["final"]["true_feasible"] == row["final"]["true_feasible"],
        }


def comparison(left, right):
    if left["final"]["true_feasible"] != right["final"]["true_feasible"]:
        return "win" if left["final"]["true_feasible"] else "loss"
    difference = left["regret"] - right["regret"]
    return "win" if difference < -1e-12 else "loss" if difference > 1e-12 else "tie"


def summarize(spec, output, cases, rows):
    groups = {}
    for profile in ["overall", *unknown.task.PROFILES]:
        selected = [row for row in rows if profile == "overall" or row["profile"] == profile]
        if not selected:
            continue
        groups[profile] = {arm: {
            "cases": len(items), "feasible": sum(row["final"]["true_feasible"] for row in items),
            "mean_regret": mean(row["regret"] for row in items),
            "objective_mse": mean(row["generalization"]["objective_mse"] for row in items),
            "constraint_mse": mean(row["generalization"]["constraint_mse"] for row in items),
            "new_simulator_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in items),
            "optimizer_seconds": sum(row["final"]["total_time_sec"] for row in items),
            "prior_preparation_seconds": sum(row["costs"].get("source_fit_seconds", 0.)
                + row["costs"].get("initial_basis_fit_seconds", 0.) for row in items),
            "total_pipeline_seconds": sum(row["final"]["total_time_sec"]
                + row["costs"].get("source_fit_seconds", 0.)
                + row["costs"].get("initial_basis_fit_seconds", 0.) for row in items),
            "objective_research_compute": sum(row.get("objective_research", {}).get("compute_cost", 0)
                                               for row in items),
            "constraint_research_compute": sum(row["costs"].get("recovery_compute", {}).get("compute_cost", 0)
                                                for row in items),
        } for arm in ARMS for items in [[r for r in selected if r["arm"] == arm]]}
    paired = {control: Counter() for control in (NATIVE, STRUCTURAL)}
    structural_vs_native = Counter()
    for case in cases:
        arms = {row["arm"]: row for row in rows if row["case_id"] == case["case_id"]}
        for control in paired:
            paired[control][comparison(arms[HEGEL], arms[control])] += 1
        structural_vs_native[comparison(arms[STRUCTURAL], arms[NATIVE])] += 1
        if len({json.dumps([arms[a]["initial_points"], arms[a]["initial_outputs"]]) for a in ARMS}) != 1:
            raise RuntimeError("the three systems did not receive identical paid initialization")
    summary = {"study": spec["experiment"], "registered_cases": len(spec["cases"]),
               "completed_cases": len(cases), "profile_results": groups,
               "hegel_paired_feasibility_first": {k: dict(v) for k, v in paired.items()},
               "structural_vs_native_feasibility_first": dict(structural_vs_native),
               "structural_prior_diagnostics": {
                   phase: {head: {
                       "selected_basis": dict(Counter(row[f"structural_prior_{phase}"]
                           ["basis_diagnostics"][head]["selected_basis"] for row in items)),
                       "coefficient_prior_applied": dict(Counter(
                           str(row[f"structural_prior_{phase}"]["basis_diagnostics"][head]
                               .get("coefficient_prior_applied", "not_recorded"))
                           for row in items)),
                   } for head in ("0", "1")}
                   for phase in ("initial", "final")
                   for items in [[row for row in rows if row["arm"] == STRUCTURAL]]},
               "new_initial_calls": 0, "new_evaluation_calls": 0,
               "new_online_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in rows),
               "timing_scope": "Newly incurred optimization wall time plus full-prior source and "
                   "initial gate fitting. Hegel's historical prefix-program learning and the "
                   "shared prefix/evaluation acquisitions are reused and excluded from this run's "
                   "wall time. total_pipeline_seconds is not end-to-end learning cost.",
               "historical_replay": {arm: {
                   "cases": len(items),
                   "same_recommendation": sum(r["historical_replay"]["same_recommendation"] for r in items),
                   "same_objective": sum(abs(r["historical_replay"]["objective_difference"]) < 1e-12 for r in items),
                   "same_feasibility": sum(r["historical_replay"]["same_feasibility"] for r in items),
               } for arm in (NATIVE, HEGEL) for items in [[r for r in rows if r["arm"] == arm]]},
               "results": rows, "scope": spec["scope"], "comparison": spec["comparison"],
               "structural_prior_data": spec["structural_prior_data"]}
    name = "summary.json" if len(cases) == len(spec["cases"]) else "development_summary.json"
    study.write(output / name, summary)
    study.write(output / name.replace("summary", "brief"),
                {key: value for key, value in summary.items() if key != "results"})
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    return summary


def run(output, selected_cases=None):
    output = Path(output).resolve()
    spec = prepare(output)
    sys.path.insert(0, spec["sc_kg_project_root"])
    modules, _ = goal.load_method(spec, "unknown_structural_prior_comparison")
    unknown.task.Environment = study.SCKGEnvironment
    study.ARMS = (HEGEL,)
    cases = [case for case in spec["cases"] if not selected_cases or case["case_id"] in selected_cases]
    if not cases or (selected_cases and set(selected_cases) != {case["case_id"] for case in cases}):
        raise ValueError("select cases from the registered nine-case roster")
    research = {case["research_id"]: study.read(
        SOURCE / "fresh_confirmation/research" / f"{case['research_id']}.json") for case in cases}
    rows = []
    for case in cases:
        saved = research[case["research_id"]]
        # Rotate execution order across cases while keeping every arm's RNG seed.
        offset = spec["cases"].index(case) % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            if arm == HEGEL:
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
            else:
                row = run_fixed_prior(spec, output, case, saved, arm)
            evaluate_saved(case, row)
            study.write(output / "runs" / f"{case['case_id']}_{arm}.json", row)
            rows.append(row)
    return summarize(spec, output, cases, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", help="Run one registered smoke case, retaining the nine-case protocol.")
    arguments = parser.parse_args()
    run(arguments.output, arguments.case)
