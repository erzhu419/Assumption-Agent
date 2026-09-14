#!/usr/bin/env python3
"""Compare frozen Hegel v0.42 with the original SC-KG optimizer."""

import argparse
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
REPLAY_CASES = (
    "research_scope_2_objective_only_change_noise_0.03",
    "research_scope_3_objective_only_change_noise_0.03",
    "objective_only_change_3_noise_0.03",
    "active_confirmation_2_mechanism_change_noise_0.03",
    "forward_1_mechanism_change_noise_0.00",
    "research_scope_3_stable_rzdt2_noise_0.00",
    "value_precision_3_mechanism_change_noise_0.03",
    "supported_exploration_2_objective_only_change_noise_0.03",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def native_modules(project_root):
    path = str(Path(project_root).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("algorithms.single_olhkg")


def initial_records(spec, case):
    source = read(Path(spec["source_study"]) / "research" / f"{case['research_id']}.json")
    state = source.get("optimizer_initial_environment_state",
        source["result"]["research_state"]["environment_state"])
    return [{"point": point, "output": [.5 * vector[0] + .5 * vector[1], vector[2]]}
        for acquisition in state["acquisitions"]
        for point, vector in zip(acquisition["integer_points"], acquisition["vector_outputs"])]


def build_vanilla_optimizer(spec, case, records, *, initialization="shared_paid32"):
    """Only the two initial-data hooks differ from the original optimizer."""
    native = native_modules(spec["sc_kg_project_root"])
    import numpy as np
    from problems.rzdt import RZDT2, StatePolicyRZDT1
    from problems.single_objective import ScalarizedProblem

    class ObjectiveOnlyChange(StatePolicyRZDT1):
        def true_objectives(self, point):
            objectives = RZDT2.true_objectives(self, point)
            return objectives[0], objectives[1], StatePolicyRZDT1.true_objectives(self, point)[2]

    class ScheduledProblem:
        def __init__(self):
            cls = RZDT2 if case["task"]["problem"] == "rzdt2" else StatePolicyRZDT1
            settings = dict(d=case["task"]["dimension"], L=100,
                sigma=case["task"]["noise"], heteroscedastic=False)
            self.base = ScalarizedProblem(cls(**settings))
            self.next_base = ScalarizedProblem((ObjectiveOnlyChange if case["profile"] ==
                "objective_only_change" else RZDT2)(**settings))
            self.global_calls = 32 if initialization == "shared_paid32" else 0
            self.actual_simulator_calls = 0
            self.simulation_log = []

        def __getattr__(self, name):
            return getattr(self.base, name)

        def simulate(self, point, rng=None):
            stage = self.global_calls + 1
            changed = (case["profile"] in ("mechanism_change", "outlier_then_change", "objective_only_change")
                and stage > 32 + case.get("change_after_new_calls", 12))
            if changed:
                self.base = self.next_base
            vector = [float(value) for value in self.base.base.simulate(point, rng)]
            outlier = case.get("observation_outlier")
            delta = (float(outlier["constraint_delta"]) if outlier and stage ==
                outlier.get("global_stage", 32 + outlier["online_call"]) else 0.)
            vector[2] += delta
            output = [float(.5 * vector[0] + .5 * vector[1]), vector[2]]
            self.global_calls += 1
            self.actual_simulator_calls += 1
            self.simulation_log.append({"global_stage": stage,
                "mechanism": case["next_problem"] if changed else case["task"]["problem"],
                "point": list(map(int, point)), "vector_outputs": vector, "output": output,
                "measurement_constraint_disturbance": delta})
            return np.asarray(output, dtype=float)

    problem = ScheduledProblem()
    settings = dict(spec["optimizer_settings"])
    config = native.SingleOLHKGConfig(N=64,
        seed=case["task"]["seed"] + spec["optimizer_seed_offset"], **settings)
    if initialization == "native_initial8":
        return native.SingleOLHKGAlgorithm(problem, config), problem
    if len(records) != 32:
        raise ValueError("The shared-data comparison requires all 32 actually paid observations")

    class PaidDataSCAlgorithm(native.SingleOLHKGAlgorithm):
        def _initial_samples(self):
            return [tuple(record["point"]) for record in records]

        def _fit_initial_belief(self, samples):
            started = perf_counter()
            for record in records:
                point = tuple(record["point"])
                output = np.asarray(record["output"], dtype=float)
                self.observations.setdefault(point, []).append(output)
                self.history.append((point, output))
            for index in range(2):
                self._rebuild_gpr_from_history(index, replay_sequential=False)
            self._configure_hvd_source_task_posterior(self.variance_model, self.gpr)
            # HVD iterates all recorded labels at each supplied control.
            unique = list(dict.fromkeys(samples))
            self.variance_model.initialize(unique, self.observations, self.gpr, self.problem)
            self._initialize_task_ensemble(unique)
            self._initialize_certification_recheck_targets(unique)
            self._last_initial_belief_timing = {"load_and_fit_paid_data": perf_counter() - started,
                "total": perf_counter() - started}

    return PaidDataSCAlgorithm(problem, replace(config, n0=32)), problem


def protocol(source, output, *, hegel_arm=None, reuse_vanilla=None):
    old = read(source / "protocol.json")
    spec = {"experiment": "frozen_v042_vs_original_sc_kg",
        "source_study": str(source.resolve()), "hegel_arm": "supported_exploration",
        "frozen_hegel_source": str((output / "baseline_source_v042").resolve()),
        "sc_kg_project_root": old["sc_kg_project_root"], "optimizer_settings": old["optimizer_settings"],
        "optimizer_seed_offset": old["optimizer_seed_offset"], "cases": old["cases"],
        "replay_cases": list(REPLAY_CASES), "total_calls_per_arm": 64,
        "primary": "Same 32 actually paid vectors, then original SC-KG for 32 online calls. "
            "Two data-loading hooks only; original default GP bases, candidate generation, MC1 acquisition, "
            "posterior updates and terminal recommendation. No Hegel learned program, adoption, "
            "hypothesis research, scope repair, objective research or numerical value refinement.",
        "initialization": "All repeated labels retained exactly once. HVD receives the unique control "
            "inventory because it itself iterates each control's full observation list. This matches "
            "the frozen Hegel initial-data loading semantics without adding cognitive decisions.",
        "supplement": "native_initial8 instantiates the original class directly, using its own "
            "8 random initial controls and all 64 physical calls. Same original settings and seeds. "
            "Changes/outliers remain tied to global physical stage 45/41, independent of initial size.",
        "scope": "All 301 cases are previously inspected development cases for Hegel. "
            "The latest 27 can be reported separately but are not unseen method confirmation. "
            "Shared-data results isolate the research layer conditional on a Hegel-chosen prefix; "
            "the native-initialization supplement does not inherit that prefix.",
        "reuse": "All 301 frozen Hegel results, 151 initial datasets and 172 evaluation datasets are reused. "
            "No existing true vanilla run matches this complete roster. Primary new cost: 301*32=9632 "
            "physical calls, zero new initial/evaluation measurements. Native-initial8 costs 64 per run.",
        "outcome": "Feasibility first; objective compared within equal feasibility status. "
            "Raw objective and feasibility transitions retained separately; all losses retained."}
    if hegel_arm is not None:
        spec.update(experiment="current_hegel_vs_original_sc_kg", hegel_arm=hegel_arm,
            frozen_hegel_source=str((source / "method_source_v043").resolve()),
            fresh_cases_unseen_at_method_freeze=True,
            scope="Current v0.43 has 301 previously seen development cases and 27 prospective cases. "
                "The Hegel method and seeds were frozen before new initialization/evaluation or optimization. "
                "Vanilla algorithms are unchanged; shared data isolates the research layer conditional on "
                "a Hegel-chosen prefix, while native-initial8 tests the complete native workflow.",
            reuse="Reuse matching completed vanilla runs by case and initialization mode. Only the new27 "
                "need shared32 and native64:864+1728=2592 new physical calls, zero new evaluation calls.")
    if hegel_arm is not None and old.get("experiment") == "terminal_research_sequence_value":
        development = sum(case["split"] == "development" for case in old["cases"])
        fresh = len(old["cases"]) - development
        spec.update(frozen_hegel_source=old["new_method"]["runtime_source"],
            scope=f"Frozen research-sequence method: {development} previously seen development cases and {fresh} prospective cases. "
                "The method and seeds were fixed before new data. Shared-data vanilla conditions on the paid Hegel prefix; native initialization is the complete original workflow.")
    if reuse_vanilla is not None:
        spec["reuse_vanilla_study"] = str(reuse_vanilla.resolve())
    return spec


def run_case(spec, output, case, initialization):
    arm = f"vanilla_{initialization}"
    path = output / "runs" / f"{case['case_id']}_{arm}.json"
    if path.exists():
        return read(path)
    if spec.get("reuse_vanilla_study"):
        original = Path(spec["reuse_vanilla_study"]) / "runs" / path.name
        if original.exists():
            result = read(original)
            result.update(comparison_origin="reused_completed_vanilla", original_result=str(original.resolve()),
                original_source_split=result["source_split"], source_split=case["split"])
            write(path, result)
            return result
    records = initial_records(spec, case) if initialization == "shared_paid32" else []
    algorithm, problem = build_vanilla_optimizer(spec, case, records, initialization=initialization)
    final = algorithm.run()
    source = Path(spec["source_study"])
    frozen = read(source / "runs" / f"{case['case_id']}_{spec['hegel_arm']}.json")
    if records and [list(x) for x in algorithm.pre_sampling_log["samples"]] != frozen["initial_points"]:
        raise RuntimeError("Vanilla initial controls differ from the frozen Hegel prefix")
    initial_outputs = [[float(v) for v in y] for _, y in algorithm.history[:algorithm.config.n0]]
    if records and initial_outputs != frozen["initial_outputs"]:
        raise RuntimeError("Vanilla did not consume exactly the same paid output vectors")
    expected_actual = 32 if records else 64
    if len(algorithm.history) != 64 or problem.actual_simulator_calls != expected_actual:
        raise RuntimeError("The native comparison did not retain the original N64 physical budget")
    evaluation_key = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}"
    evaluation = read(source / "evaluation" / f"{evaluation_key}.json")
    controls = evaluation["integer_points"]
    prediction = [[float(model.posterior_mean(point)) for point in controls] for model in algorithm.gpr]
    result = {"case_id": case["case_id"], "profile": case["profile"], "task": case["task"],
        "source_split": case["split"], "arm": arm, "config": asdict(algorithm.config),
        "class": f"{type(algorithm).__module__}.{type(algorithm).__qualname__}",
        "basis_maps": [None if model.basis_map is None else type(model.basis_map).__name__ for model in algorithm.gpr],
        "initial_points": algorithm.pre_sampling_log["samples"], "initial_outputs": initial_outputs,
        "simulations": problem.simulation_log,
        "actions": [{key: row.get(key) for key in ("stage", "x_selected", "Y_observed", "mu_before", "action_kind",
            "n_candidates", "exact_kg_raw_selected", "score_selected", "exact_kg_mc_samples", "t_kg_compute")}
            for row in algorithm.iteration_log],
        "costs": {"total_simulator_calls": 64, "actual_new_simulator_calls": problem.actual_simulator_calls,
            "reused_initial_calls": 32 if records else 0, "actual_new_initial_calls": 0 if records else algorithm.config.n0,
            "actual_new_online_calls": 32 if records else 64-algorithm.config.n0,
            "new_evaluation_calls": 0, "evaluation_prediction_rows": 2*len(controls),
            "native_kg_scored_actions": sum(row.get("n_candidates",0) for row in algorithm.iteration_log),
            "native_kg_seconds": sum(row.get("t_kg_compute",0.) for row in algorithm.iteration_log),
            "hegel_research_compute": 0},
        "generalization": {"objective_gp_true_mse": mean((a-b)**2 for a,b in zip(prediction[0], evaluation["true_objective_means"])),
            "constraint_gp_true_mse": mean((a-b)**2 for a,b in zip(prediction[1], evaluation["true_constraint_means"]))},
        "final": {key: final.get(key) for key in ("x_recommended", "true_objective", "true_constraint_mean",
            "true_feasible", "n_simulations", "total_time_sec")}}
    write(path, result)
    print(json.dumps({"case": case["case_id"], "arm": arm, "final": result["final"]}), flush=True)
    return result


def summarize(spec, output, results, name):
    pairs = []
    source = Path(spec["source_study"])
    for vanilla in results:
        hegel = read(source / "runs" / f"{vanilla['case_id']}_{spec['hegel_arm']}.json")
        old, new = vanilla["final"], hegel["final"]
        delta = new["true_objective"] - old["true_objective"]
        raw = "improved" if delta < -1e-9 else "worse" if delta > 1e-9 else "unchanged"
        if old["true_feasible"] != new["true_feasible"]:
            outcome = "improved" if new["true_feasible"] else "worse"
        else:
            outcome = raw if new["true_feasible"] else "unchanged"
        recovery = hegel["costs"].get("recovery_compute", {}).get("compute_cost", 0)
        objective = (hegel.get("objective_research") or {}).get("compute_cost", 0)
        kg_rows = hegel["costs"].get("exact_kg_stages")
        case = next(c for c in spec["cases"] if c["case_id"] == vanilla["case_id"])
        evaluation_key = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}"
        evaluation = read(source / "evaluation" / f"{evaluation_key}.json")
        prediction = hegel["frozen_evaluation_predictions"]
        hegel_mse = {"objective_gp_true_mse": mean((a-b)**2 for a,b in zip(
            prediction["objective_gp"], evaluation["true_objective_means"])),
            "constraint_gp_true_mse": mean((a-b)**2 for a,b in zip(
            prediction["active_gp"], evaluation["true_constraint_means"]))}
        pairs.append({"case_id": vanilla["case_id"], "profile": vanilla["profile"], "noise": vanilla["task"]["noise"],
            "source_split": case["split"], "vanilla_arm": vanilla["arm"],
            "objective": [old["true_objective"], new["true_objective"]],
            "feasible": [old["true_feasible"], new["true_feasible"]],
            "raw_objective_outcome": raw, "feasibility_first_outcome": outcome,
            "gp_true_mse": {key: [vanilla["generalization"][key],value] for key,value in hegel_mse.items()},
            "hegel_search_compute": recovery + objective,
            "hegel_initial_research_compute": hegel["costs"].get("research_compute", 0),
            "hegel_value_precision_evaluations": hegel["costs"].get("objective_diagnostics", {}).get(
                "value_precision", {}).get("evaluations", 0),
            "native_kg_seconds": [vanilla["costs"]["native_kg_seconds"],
                None if kg_rows is None else sum(row.get("t_kg_compute") or 0 for row in kg_rows)],
            "total_seconds": [old["total_time_sec"],new["total_time_sec"]]})
    def aggregate(rows):
        both = [r for r in rows if all(r["feasible"])]
        relative = [(r["objective"][0]-r["objective"][1])/abs(r["objective"][0]) for r in both]
        return {"cases": len(rows), "feasibility_first": {key: sum(r["feasibility_first_outcome"]==key for r in rows)
                for key in ("improved","unchanged","worse")},
            "raw_objective": {key: sum(r["raw_objective_outcome"]==key for r in rows) for key in ("improved","unchanged","worse")},
            "feasible": [sum(r["feasible"][i] for r in rows) for i in (0,1)],
            "feasibility_flips": {"gained_by_hegel":sum(r["feasible"]==[False,True] for r in rows),
                "lost_by_hegel":sum(r["feasible"]==[True,False] for r in rows)},
            "both_infeasible_cases":sum(r["feasible"]==[False,False] for r in rows),
            "both_feasible": {"cases":len(both),
                "mean_objective":[mean(r["objective"][i] for r in both) if both else None for i in (0,1)],
                "mean_relative_improvement":mean(relative) if relative else None,
                "median_relative_improvement":median(relative) if relative else None},
            "gp_loss_counts": {key: {
                "improved":sum(r["gp_true_mse"][key][1] < r["gp_true_mse"][key][0]-1e-9 for r in rows),
                "unchanged":sum(abs(r["gp_true_mse"][key][1]-r["gp_true_mse"][key][0]) <= 1e-9 for r in rows),
                "worse":sum(r["gp_true_mse"][key][1] > r["gp_true_mse"][key][0]+1e-9 for r in rows)}
                for key in ("objective_gp_true_mse","constraint_gp_true_mse")},
            "mean_objective": [mean(r["objective"][i] for r in rows) if rows else None for i in (0,1)]}
    group_keys = sorted({(r["profile"],r["noise"]) for r in pairs})
    cost_fields = ("actual_new_simulator_calls", "actual_new_initial_calls", "actual_new_online_calls",
        "reused_initial_calls", "new_evaluation_calls", "evaluation_prediction_rows", "native_kg_scored_actions", "native_kg_seconds")
    actual = [r for r in results if r.get("comparison_origin") != "reused_completed_vanilla"]
    summary = {"comparison_order": ["vanilla", spec["hegel_arm"]],
        "all_cases_previously_seen": not spec.get("fresh_cases_unseen_at_method_freeze", False),
        "overall": aggregate(pairs), "source_latest27": aggregate([r for r in pairs if r["source_split"] != "development"]),
        "groups": [{"profile":p,"noise":n,**aggregate([r for r in pairs if (r['profile'],r['noise'])==(p,n)])} for p,n in group_keys],
        "actual_new_costs": {key:sum(r["costs"][key] for r in actual) for key in cost_fields},
        "reused_vanilla_runs": len(results)-len(actual),
        "matched_native_kg": {"cases": sum(r["native_kg_seconds"][1] is not None for r in pairs),
            "seconds": [sum(r["native_kg_seconds"][i] for r in pairs if r["native_kg_seconds"][1] is not None) for i in (0,1)],
            "hegel_missing_cases": sum(r["native_kg_seconds"][1] is None for r in pairs)},
        "reused_hegel_search_compute": sum(r["hegel_search_compute"] for r in pairs),
        "reused_hegel_initial_research_compute": sum(r["hegel_initial_research_compute"] for r in pairs),
        "reused_hegel_value_precision_evaluations": sum(r["hegel_value_precision_evaluations"] for r in pairs),
        "pairs":pairs}
    write(output / f"{name}_summary.json", summary)
    print(json.dumps({"overall":summary["overall"],"actual_new_costs":summary["actual_new_costs"]}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("plan","replay","shared_all","native_latest27"),default="plan")
    parser.add_argument("--case",action="append")
    parser.add_argument("--hegel-arm")
    parser.add_argument("--reuse-vanilla-study",type=Path)
    args=parser.parse_args()
    spec=protocol(args.source_study,args.output,hegel_arm=args.hegel_arm,reuse_vanilla=args.reuse_vanilla_study)
    path=args.output/"protocol.json"
    if path.exists() and read(path)!=spec:
        raise ValueError("The saved vanilla protocol differs; retain it instead of overwriting")
    write(path,spec)
    if args.phase=="plan":
        print(json.dumps({"cases":len(spec["cases"]),"replay":list(REPLAY_CASES)}));return
    cases=spec["cases"]
    if args.case:
        unknown=set(args.case)-{c["case_id"] for c in cases}
        if unknown:raise ValueError(f"Unknown frozen cases: {sorted(unknown)}")
        cases=[c for c in cases if c["case_id"] in args.case]
    elif args.phase=="replay":
        cases=[c for c in cases if c["case_id"] in REPLAY_CASES]
    elif args.phase=="native_latest27":
        cases=[c for c in cases if c["split"]!="development"]
    mode="native_initial8" if args.phase=="native_latest27" else "shared_paid32"
    results=[run_case(spec,args.output,c,mode) for c in cases]
    summarize(spec,args.output,results,args.phase)


if __name__=="__main__":
    main()
