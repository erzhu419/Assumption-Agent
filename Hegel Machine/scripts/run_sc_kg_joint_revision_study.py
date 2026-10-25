#!/usr/bin/env python3
"""Frozen common-misfit revision comparison against the actual 0.31 adapter."""

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import random
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
from hegel_machine.cognition.models import Probe, ResearchConfig
from hegel_machine.cognition.sc_kg_environment import SCKGEnvironment


OLD_SEED = 1043010067
FORWARD_SEEDS = (1143010079, 1143110091, 1143210103)
NOISE_LEVELS = (0.0, 0.03)
ARMS = ("frozen_v031_auto", "joint_revision")
CHANGE_AFTER = 12
INITIAL_COMPLETION_SEED_OFFSET = 7_000_019


def read(path):
    return json.loads(Path(path).read_text())


def write(path, body):
    write_json_checkpoint(path, body, indent=2, ensure_ascii=False, allow_nan=True)


def make_protocol(source_change, output):
    source = read(source_change / "protocol.json")
    cases = [{"case_id": "development_change_noise_0.00", "split": "development",
        "profile": "mechanism_change", "task": source["task"],
        "next_problem": "rzdt2", "research_id": "development_state_noise_0.00",
        "existing_research": str((source_change / "research.json").resolve())}]
    for index, seed in enumerate(FORWARD_SEEDS, 1):
        for noise in NOISE_LEVELS:
            for profile, problem in (("mechanism_change", "state_policy_rzdt1"),
                                     ("stable_state_policy", "state_policy_rzdt1"),
                                     ("stable_rzdt2", "rzdt2")):
                cases.append({"case_id": f"forward_{index}_{profile}_noise_{noise:.2f}",
                    "split": "forward", "profile": profile,
                    "task": {"problem": problem, "dimension": 3, "seed": seed, "noise": noise},
                    "next_problem": "rzdt2" if profile == "mechanism_change" else problem,
                    "research_id": f"forward_{index}_{problem}_noise_{noise:.2f}"})
    baseline_dir = output / "baseline_source"
    for filename in ("sc_kg_environment.py", "prior_adoption.py"):
        if not (baseline_dir / filename).exists():
            raise FileNotFoundError(f"the pre-edit 0.31 source snapshot is required: {baseline_dir / filename}")
    return {"experiment": "actual_joint_model_revision", "cases": cases, "arms": list(ARMS),
        "source_change": str(source_change.resolve()),
        "old_recorded_summary": str((source_change / "summary.json").resolve()),
        "frozen_baseline_source": str(baseline_dir.resolve()),
        "baseline_scope": "The complete 0.31 SC adapter/initializer and prior-adoption policy were copied before the joint-revision edits. Baseline runs import these files, not the changed initializer. The existing old-seed summary remains a separate recorded reference.",
        "sc_kg_project_root": source["sc_kg_project_root"],
        "research_config": source["research_config"],
        "total_calls_per_arm": 64, "change_after_new_calls": CHANGE_AFTER,
        "optimizer_seed_offset": source["optimizer_seed"] - OLD_SEED,
        "optimizer_settings": source["optimizer_settings"], "policy": source["policy"],
        "revision": {"minimum_error": 1e-7, "batch_size": 4, "compute_budget": 20000,
            "seed_offset": 5_000_011,
            "evidence": "Both models' previously measured error and observation noise define a frozen common-misfit reference. Four consecutive anomalous actual observations reopen the world explanation; eight public coverage measurements fit candidates and four unseen measurements validate them. A supported candidate replaces the active explanation and scopes all GPs to the new segment. The controller receives no mechanism label, change time, or simulator truth."},
        "evaluation": {"count": 128, "seed_offset": 6_000_017,
            "rule": "Public evaluation coordinates and all final Program/GP predictions are frozen after each optimizer run. After every selected run is complete, a separate actual simulator stream supplies shared measurements and evaluator-only true constraint means. Evaluation calls never enter the controller and are reported separately."},
        "accounting": "Research acquires 32 actual full objective/constraint vectors once per distinct initial task and shares them across both arms and same-task stable/change profiles. All loaded calls count toward each arm's N64. New revision questions replace ordinary optimizer calls inside the same remaining 32 calls; full history is retained. Historical reuse, new acquisition, extra model compute, and independent evaluation calls are separate.",
        "metrics": ["post-change pre-measurement active/default/learned forecast MSE",
            "forecast MSE after accepted replacement", "final Program and active GP generalization MSE",
            "final recommendation objective and feasibility", "actual call and compute costs",
            "common misfit, question, fresh validation, and scope adoption sequence",
            "stable-control false reopening and replacement"],
        "scope": "The old development case is checked first. Three new seeds and both noise levels were selected before executing this protocol. Mechanism changes and each stable control are reported separately by noise; unresolved revision and performance regressions remain results."}


def complete_initial_data(spec, case, saved, path):
    """Preserve early research completion and acquire the protocol's initial 32."""
    if saved.get("optimizer_initial_environment_state"):
        return saved
    research_calls = saved["result"]["costs"]["observation_cost"]
    missing = 32 - research_calls
    if missing < 0:
        raise ValueError("research exceeds the frozen initial data budget")
    environment = SCKGEnvironment(spec["sc_kg_project_root"], **case["task"])
    environment.restore(saved["result"]["research_state"]["environment_state"])
    if missing:
        rng = random.Random(case["task"]["seed"] + INITIAL_COMPLETION_SEED_OFFSET)
        columns = {}
        for name in environment.spec.feature_names:
            values = [(index + rng.random()) / missing for index in range(missing)]
            rng.shuffle(values)
            columns[name] = values
        controls = tuple({name: values[index] for name, values in columns.items()}
            for index in range(missing))
        environment.observe(Probe("protocol_initial_data_completion", controls))
    saved["supplemental_initial_calls"] = missing
    saved["optimizer_initial_environment_state"] = environment.snapshot()
    saved["actual_new_initial_calls"] = saved["actual_new_research_calls"] + missing
    write(path, saved)
    return saved


def acquire_research(spec, output, cases):
    reports = {}
    for case in cases:
        key = case["research_id"]
        if key in reports:
            continue
        path = output / "research" / f"{key}.json"
        saved = read(path) if path.exists() else None
        if saved and saved.get("finished"):
            reports[key] = complete_initial_data(spec, case, saved, path)
            continue
        if case.get("existing_research"):
            original = read(case["existing_research"])
            if original["task"] != case["task"]:
                raise ValueError("historical research task differs from the frozen task")
            saved = {**original, "finished": True, "research_id": key,
                "research_origin": case["existing_research"], "actual_new_research_calls": 0}
            write(path, saved)
            reports[key] = complete_initial_data(spec, case, saved, path)
            continue
        environment = SCKGEnvironment(spec["sc_kg_project_root"], **case["task"])
        library = KnowledgeLibrary.from_dict(saved["library"]) if saved else KnowledgeLibrary()
        config = replace(ResearchConfig(**spec["research_config"]), seed=case["task"]["seed"])

        def checkpoint(report):
            write(path, {"research_id": key, "task": case["task"], "result": report,
                "library": library.to_dict(), "finished": False,
                "research_origin": "new_public_measurements",
                "actual_new_research_calls": report["costs"]["observation_cost"]})

        report = MetaPriorResearch(config, library).run(environment,
            research_state=saved["result"]["research_state"] if saved else None,
            on_checkpoint=checkpoint)
        checkpoint(report)
        saved = read(path)
        saved["finished"] = True
        write(path, saved)
        reports[key] = complete_initial_data(spec, case, saved, path)
        print(json.dumps({"research": key, "status": report["status"], "costs": report["costs"]}), flush=True)
    return reports


def load_baseline(spec):
    modules = {}
    for name in ("sc_kg_environment", "prior_adoption"):
        module_name = f"hegel_machine.cognition._joint_revision_baseline_{name}_v031"
        loader = importlib.util.spec_from_file_location(module_name,
            Path(spec["frozen_baseline_source"]) / f"{name}.py")
        module = importlib.util.module_from_spec(loader)
        sys.modules[module_name] = module
        loader.loader.exec_module(module)
        modules[name] = module
    return modules


def controlled_problem(base_class, environment, program, case):
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
            changed = case["profile"] == "mechanism_change" and self.simulator_calls >= CHANGE_AFTER
            if changed and self.simulator_calls == CHANGE_AFTER:
                self.base = ScalarizedProblem(self.next_problem)
            vector = [float(value) for value in self.base.base.simulate(point, rng)]
            disturbance = case.get("observation_outlier")
            applied_disturbance = (float(disturbance["constraint_delta"])
                if disturbance and self.simulator_calls + 1 == disturbance["online_call"] else 0.0)
            vector[2] += applied_disturbance
            output = [float(self.base.weights[0] * vector[0] + self.base.weights[1] * vector[1]), vector[2]]
            self.simulator_calls += 1
            self.simulation_log.append({"new_call": self.simulator_calls,
                "mechanism": case["next_problem"] if changed else case["task"]["problem"],
                "point": list(map(int, point)), "vector_outputs": vector, "output": output})
            if applied_disturbance:
                self.simulation_log[-1]["measurement_constraint_disturbance"] = applied_disturbance
            return np.asarray(output, dtype=float)

    return ControlledProblem()


def evaluation_controls(spec, case):
    rng = random.Random(case["task"]["seed"] + spec["evaluation"]["seed_offset"])
    return [[rng.randrange(101) for _ in range(case["task"]["dimension"])]
            for _ in range(spec["evaluation"]["count"])]


def optimize(spec, output, cases, research, *, arm_modules=None):
    from hegel_machine.cognition import sc_kg_environment as current
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption

    if arm_modules is None:
        baseline = load_baseline(spec)
        arm_modules = {
            ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption, None),
            "joint_revision": (current, PriorAdoption, JointModelRevision),
        }
    results = []
    for case in cases:
        report = research[case["research_id"]]["result"]
        payload = report.get("program") or (report["research_state"].get("incumbent") or {}).get("candidate", {}).get("program")
        source = research[case["research_id"]]
        if payload is None or report["costs"]["observation_cost"] + source.get("supplemental_initial_calls", 0) != 32:
            raise ValueError("each frozen case needs exactly 32 initial paid rows and an executable candidate")
        for arm in ARMS:
            path = output / "runs" / f"{case['case_id']}_{arm}.json"
            if path.exists():
                results.append(read(path))
                continue
            module, policy_class, recovery_class = arm_modules[arm]
            environment = module.SCKGEnvironment(spec["sc_kg_project_root"], **case["task"])
            environment.restore(source.get("optimizer_initial_environment_state",
                report["research_state"]["environment_state"]))
            adapter = controlled_problem(module.HegelProgramProblem, environment, Program.from_dict(payload), case)
            from algorithms.single_olhkg import SingleOLHKGConfig
            config = SingleOLHKGConfig(N=spec["total_calls_per_arm"],
                seed=case["task"]["seed"] + spec["optimizer_seed_offset"], **spec["optimizer_settings"])
            adoption = policy_class(minimum_gain=spec["policy"]["minimum_gain"],
                batch_size=spec["policy"]["minimum_new_observations"])
            kwargs = {"adoption_policy": adoption}
            recovery = None
            if recovery_class is not None:
                recovery = recovery_class(environment.spec,
                    minimum_error=spec["revision"]["minimum_error"],
                    batch_size=spec["revision"]["batch_size"],
                    seed=case["task"]["seed"] + spec["revision"]["seed_offset"],
                    compute_budget=spec["revision"]["compute_budget"])
                kwargs["recovery"] = recovery
            algorithm = module.initialize_sc_kg_optimizer(adapter, config, **kwargs)
            final = algorithm.run()
            final_certificate = algorithm._terminal_certificate_components(
                algorithm.gpr, algorithm.variance_model, [final["x_recommended"]],
                observations=algorithm.observations)
            final_certificate = {key: float(final_certificate[key][0]) for key in
                ("objective", "mu_con", "epistemic", "aleatoric", "margin")}
            learned_evaluations = adapter.basis.evaluations
            controls = evaluation_controls(spec, case)
            # No evaluation targets exist yet; these are final frozen predictions.
            frozen_prediction = {"integer_points": controls,
                "program": [adapter.basis.features(point)[0] for point in controls],
                "active_gp": [float(algorithm.gpr[1].posterior_mean(point)) for point in controls],
                "objective_gp": [float(algorithm.gpr[0].posterior_mean(point)) for point in controls]}
            precision_events = getattr(algorithm, "objective_diagnostics", {}).get("value_precision", {}).get("rounds", [])
            precision_annotations = {event["stage"] - 1: {
                "selection_semantics": event["selection_semantics"],
                "selected_raw_value": event["selected_raw_value"],
                "value_precision_status": event["status"],
                "value_precision_evaluations": event["evaluations"],
                "value_precision_seconds": event["seconds"],
                "initial_mc1_raw_min": min(event["initial_raw"]),
                "initial_mc1_raw_max": max(event["initial_raw"]),
                "initial_mc1_selected_point": event["candidate_points"][max(range(len(event["initial_clipped"])),
                    key=event["initial_clipped"].__getitem__)]} for event in precision_events}
            sequence_events = getattr(algorithm, "objective_diagnostics", {}).get("research_value", {}).get("rounds", [])
            for event in sequence_events:
                annotation = precision_annotations.setdefault(event["stage"] - 1, {})
                annotation.update(research_value_status=event["status"],
                    research_value_selected_point=event.get("selected_point"),
                    research_value_baseline_point=event["baseline_point"],
                    research_value_planned_calls=event.get("planned_calls"),
                    research_value_evaluations=event["evaluations"], research_value_seconds=event["seconds"])
                if event.get("selected_point") is not None:
                    annotation["selection_semantics"] = "policy_indicator_execute_first_observation_then_replan"
                    if "selected_raw_value" in annotation:
                        annotation["single_observation_selected_raw_value"] = annotation.pop("selected_raw_value")
                    selected_plan = next(plan for plan in event["plan_statistics"] if
                        plan["point"] == event["selected_point"] and plan["planned_calls"] == event["planned_calls"])
                    annotation["selected_sequence_mean_total"] = selected_plan["mean_total"]
                    annotation["selected_sequence_mean_per_call"] = selected_plan["mean_per_call"]
            result = {"case_id": case["case_id"], "split": case["split"],
                "profile": case["profile"], "task": case["task"], "arm": arm,
                "research_status": report["status"], "research_family": report.get("family_id"),
                "initial_program": payload, "final_program": adapter.basis.program.to_dict(),
                "costs": {"research_simulator_calls": report["costs"]["observation_cost"],
                    "supplemental_initial_calls": source.get("supplemental_initial_calls", 0),
                    "total_initial_simulator_calls": adapter.research_simulator_calls,
                    "research_compute": report["costs"]["compute_cost"],
                    "optimizer_new_simulator_calls": adapter.simulator_calls,
                    "total_simulator_calls": adapter.research_simulator_calls + adapter.simulator_calls,
                    "adoption_compute": algorithm.adoption_diagnostics,
                    "recovery_compute": recovery.to_dict().get("costs", {}) if recovery else {},
                    "recovery_diagnostics": getattr(algorithm, "recovery_diagnostics", {}),
                    "objective_diagnostics": getattr(algorithm, "objective_diagnostics", {}),
                    "exact_kg_stages": [{**{key: row.get(key) for key in ("stage", "n_candidates",
                        "exact_kg_active_action_count", "exact_kg_mc_samples", "t_kg_compute",
                        "exact_kg_time_clone_mean", "exact_kg_time_predictive_sample_mean",
                        "exact_kg_time_joint_update_mean", "exact_kg_time_robust_terminal_mean",
                        "exact_kg_raw_min", "exact_kg_raw_max", "exact_kg_raw_selected",
                        "score_selected", "x_selected")}, **precision_annotations.get(row["stage"], {})} for row in algorithm.iteration_log],
                    "learned_basis_evaluations": learned_evaluations,
                    "evaluation_prediction_rows": len(controls),
                    "objective_evaluation_prediction_rows": len(controls)},
                "final_recommendation_certificate": final_certificate,
                "postrun_certificate_prediction_rows": 4,
                "loaded_observation_count": adapter.loaded_observation_count,
                "optimizer_history_count": len(algorithm.history),
                "initial_points": algorithm.pre_sampling_log["samples"],
                "initial_outputs": [[float(value) for value in vector]
                    for _, vector in algorithm.history[:32]],
                "adoption_policy": adoption.to_dict(), "adoption_events": algorithm.adoption_events,
                "recovery": recovery.to_dict() if recovery else None,
                "scope_revisions": getattr(algorithm, "scope_revisions", []),
                "observation_repairs": getattr(algorithm, "observation_repairs", []),
                "rejected_constraint_indices": sorted(getattr(algorithm, "rejected_constraint_indices", set())),
                "objective_research": (algorithm.objective_research.to_dict()
                    if getattr(algorithm, "objective_research", None) is not None else None),
                "objective_scope_revisions": getattr(algorithm, "objective_scope_revisions", []),
                "joint_evidence": getattr(algorithm, "joint_evidence", None),
                "simulations": adapter.simulation_log, "frozen_evaluation_predictions": frozen_prediction,
                "active_constraint_basis": algorithm.active_constraint_basis,
                "actions": [{**{key: row.get(key) for key in ("stage", "x_selected", "Y_observed",
                    "mu_before", "action_kind", "recovery_phase")}, **precision_annotations.get(row["stage"], {})} for row in algorithm.iteration_log],
                "final": {key: final.get(key) for key in ("x_recommended", "true_objective",
                    "true_constraint_mean", "true_feasible", "n_simulations", "total_time_sec")}}
            if result["costs"]["total_simulator_calls"] != 64 or len(algorithm.history) != 64:
                raise RuntimeError("revision must preserve the full paid N64 history")
            write(path, result)
            results.append(result)
            print(json.dumps({"case": case["case_id"], "arm": arm,
                "revision_phase": result["recovery"]["phase"] if recovery else None,
                "scope_revisions": len(result["scope_revisions"]), "final": result["final"]}), flush=True)
    return results


def evaluate(spec, output, cases, results, *, environment_factory=None):
    environment_factory = environment_factory or SCKGEnvironment
    evaluations = {}
    for case in cases:
        # Stable RZDT2 and switched cases share final-task evaluation measurements.
        key = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}"
        path = output / "evaluation" / f"{key}.json"
        if key in evaluations:
            continue
        if path.exists():
            saved = read(path)
            # Historical files retain their actual measurements. Only an evaluator-
            # only deterministic objective can be added; missing GP forecasts stay missing.
            if "true_objective_means" not in saved:
                environment = environment_factory(spec["sc_kg_project_root"], **saved["task"])
                from problems.single_objective import ScalarizedProblem
                scalar = ScalarizedProblem(environment._problem)
                saved["true_objective_means"] = [scalar.true_objective(point) for point in saved["integer_points"]]
                saved["objective_evaluation_origin"] = "postrun_deterministic_truth_only_no_new_measurements"
                write(path, saved)
            evaluations[key] = saved
            continue
        task = {**case["task"], "problem": case["next_problem"],
            "seed": case["task"]["seed"] + spec["evaluation"]["seed_offset"]}
        environment = environment_factory(spec["sc_kg_project_root"], **task)
        points = evaluation_controls(spec, case)
        rows = []
        for start in range(0, len(points), 64):
            controls = tuple({name: point[index] / 100.0 for index, name in enumerate(environment.spec.feature_names)}
                for point in points[start:start + 64])
            rows.extend(environment.observe(Probe(f"postrun_evaluation_{start}", controls)))
        # Evaluator-only truth is obtained after both runs have frozen their outputs.
        true_targets = [float(environment._problem.true_objectives(point)[2]) for point in points]
        from problems.single_objective import ScalarizedProblem
        scalar = ScalarizedProblem(environment._problem)
        vectors = [vector for acquisition in environment.acquisitions for vector in acquisition["vector_outputs"]]
        saved = {"integer_points": points, "measured_targets": [row.target for row in rows],
            "true_constraint_means": true_targets, "simulator_calls": len(rows), "task": task,
            "measured_objectives": [float(scalar.weights[0] * vector[0] + scalar.weights[1] * vector[1]) for vector in vectors],
            "true_objective_means": [scalar.true_objective(point) for point in points],
            "objective_evaluation_origin": "same_actual_full_vector_measurements"}
        write(path, saved)
        evaluations[key] = saved
    for result in results:
        case = next(case for case in cases if case["case_id"] == result["case_id"])
        key = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}"
        saved = evaluations[key]
        prediction = result["frozen_evaluation_predictions"]
        if prediction["integer_points"] != saved["integer_points"]:
            raise ValueError("frozen forecasts and shared measurements use different controls")
        result["generalization"] = {f"{model}_{target}_mse": mean((pred - truth) ** 2
                for pred, truth in zip(prediction[model], saved[target]))
            for model in ("program", "active_gp")
            for target in ("measured_targets", "true_constraint_means")}
        result["generalization"]["objective_gp_prediction_available"] = "objective_gp" in prediction
        if "objective_gp" in prediction:
            for target in ("measured_objectives", "true_objective_means"):
                if target in saved:
                    result["generalization"][f"objective_gp_{target}_mse"] = mean(
                        (pred - truth) ** 2 for pred, truth in zip(prediction["objective_gp"], saved[target]))
    return evaluations


def phase_metrics(result):
    events = result["adoption_events"]
    revisions = result["scope_revisions"]
    accepted_stages = [row.get("stage", row.get("accepted_stage")) for row in revisions]
    accepted_stages = [stage for stage in accepted_stages if stage is not None]
    phases = {"first_12_new_calls": events[:CHANGE_AFTER], "last_20_new_calls": events[CHANGE_AFTER:]}
    if accepted_stages:
        phases["after_first_accepted_replacement"] = [row for row in events if row["stage"] > min(accepted_stages)]
    metrics = {}
    # SC action.stage is the count before measurement; controller.stage is after it.
    objective_forecasts = {row["stage"] + 1: (row["mu_before"][0], row["Y_observed"][0])
        for row in result["actions"] if row.get("mu_before") is not None and row.get("Y_observed") is not None}
    for name, rows in phases.items():
        metrics[name] = {"rows": len(rows), **({f"{model}_mse": mean(
            (row[f"{row['selected_before'] if model == 'active' else model}_mean"] - row["target"]) ** 2
            for row in rows) for model in ("default", "learned", "active")} if rows else {})}
        objective_rows = [objective_forecasts[row["stage"]] for row in rows if row["stage"] in objective_forecasts]
        metrics[name]["objective_rows"] = len(objective_rows)
        if objective_rows:
            metrics[name]["objective_mse"] = mean((pred - actual) ** 2 for pred, actual in objective_rows)
            metrics[name]["objective_mae"] = mean(abs(pred - actual) for pred, actual in objective_rows)
    return metrics


def summarize(spec, output, cases, research, results, evaluations):
    comparisons = []
    for result in results:
        result["phase_prediction_error"] = phase_metrics(result)
    for case in cases:
        arms = {row["arm"]: row for row in results if row["case_id"] == case["case_id"]}
        old, new = (arms[arm] for arm in ARMS)
        comparisons.append({"case_id": case["case_id"], "profile": case["profile"],
            "split": case["split"], "noise": case["task"]["noise"],
            "same_initial_points": old["initial_points"] == new["initial_points"],
            "same_initial_outputs": old["initial_outputs"] == new["initial_outputs"],
            "accepted_replacements": len(new["scope_revisions"]),
            "program_changed": new["initial_program"] != new["final_program"],
            "objective_delta": new["final"]["true_objective"] - old["final"]["true_objective"],
            "active_postchange_mse_delta": new["phase_prediction_error"]["last_20_new_calls"]["active_mse"] - old["phase_prediction_error"]["last_20_new_calls"]["active_mse"],
            "program_generalization_mse_delta": new.get("generalization", {}).get("program_true_constraint_means_mse", 0.) - old.get("generalization", {}).get("program_true_constraint_means_mse", 0.)})
    aggregate = []
    groups = list(dict.fromkeys((case["split"], case["profile"], case["task"]["noise"]) for case in cases))
    for split, profile, noise in groups:
        for arm in ARMS:
            rows = [row for row in results if (row["split"], row["profile"], row["task"]["noise"], row["arm"]) == (split, profile, noise, arm)]
            aggregate.append({"split": split, "profile": profile, "noise": noise, "arm": arm,
                "runs": len(rows), "feasible": sum(row["final"]["true_feasible"] for row in rows),
                "mean_objective": mean(row["final"]["true_objective"] for row in rows),
                "accepted_replacements": sum(len(row["scope_revisions"]) for row in rows),
                "mean_active_last20_forecast_mse": mean(row["phase_prediction_error"]["last_20_new_calls"]["active_mse"] for row in rows),
                "mean_final_program_true_mse": mean(row["generalization"]["program_true_constraint_means_mse"] for row in rows),
                "mean_final_gp_true_mse": mean(row["generalization"]["active_gp_true_constraint_means_mse"] for row in rows)})
    old_reference = read(spec["old_recorded_summary"])
    summary = {"protocol": spec, "selected_cases": [case["case_id"] for case in cases],
        "execution_amendment": read(output / "initial_data_completion_amendment.json"),
        "comparisons": comparisons, "aggregate": aggregate, "results": results,
        "actual_new_research_calls": sum(row["actual_new_research_calls"] for row in research.values()),
        "actual_supplemental_initial_calls": sum(row.get("supplemental_initial_calls", 0) for row in research.values()),
        "historically_reused_research_calls": sum(row["result"]["costs"]["observation_cost"] for row in research.values() if row["research_origin"] != "new_public_measurements"),
        "actual_new_optimizer_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in results),
        "independent_evaluation_calls": sum(row["simulator_calls"] for row in evaluations.values()),
        "old_recorded_reference": {key: old_reference[key] for key in ("phase_prediction_error", "final", "adopted_at_stages", "withdrawn_at_stages")}}
    filename = "summary.json" if len(cases) == len(spec["cases"]) else "development_summary.json"
    write(output / filename, summary)
    print(json.dumps({"aggregate": aggregate, "output": str(output / filename)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-change", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "optimize", "evaluate", "all"), default="plan")
    parser.add_argument("--case", action="append", help="Execute a named frozen case, without changing the roster.")
    args = parser.parse_args()
    spec = make_protocol(args.source_change, args.output)
    path = args.output / "protocol.json"
    if path.exists() and read(path) != spec:
        raise ValueError("the saved joint-revision protocol differs")
    write(path, spec)
    amendment = {"recorded_date": "2026-09-10", "reason": "The three stable dimension-three RZDT2/noise0 research sessions completed after 20 measurements. Their optimizer arms had not run or been evaluated when this amendment was made. The original matched design requires 32 initial records and 32 online calls in both arms.",
        "rule": "Retain the original research result, candidate, status and 20-row costs. Acquire 12 additional actual full-vector measurements using independent Latin coverage over public normalized controls, without refitting the research candidate. Save these as supplemental_initial_calls and optimizer_initial_environment_state. Both optimizers load the same total 32 rows; all calls remain charged inside N64.",
        "completion_seed_offset": INITIAL_COMPLETION_SEED_OFFSET,
        "algorithm": "For each public feature, jitter every one of the remaining-count equal-width strata, shuffle with random.Random(task_seed+offset), then pair columns by row index.",
        "validation_correction": "Before continuing the forward roster, candidate acceptance was corrected to require both existing training evidence and the fresh validation batch to satisfy the same already-frozen error threshold. The completed development and forward1/noise0 change cases already satisfy both conditions; the completed stable StatePolicy case proposed no replacement. Their actual results are retained."}
    amendment_path = args.output / "initial_data_completion_amendment.json"
    if amendment_path.exists() and read(amendment_path) != amendment:
        raise ValueError("the recorded initial-data amendment differs")
    write(amendment_path, amendment)
    cases = [case for case in spec["cases"] if not args.case or case["case_id"] in args.case]
    if not cases or (args.case and set(args.case) - {case["case_id"] for case in cases}):
        raise ValueError("case selection must name frozen cases")
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "cases": len(spec["cases"]),
            "runs": len(spec["cases"]) * len(ARMS), "selected": len(cases)}))
        return
    research = (acquire_research(spec, args.output, cases) if args.phase in ("research", "all") else
        {case["research_id"]: read(args.output / "research" / f"{case['research_id']}.json") for case in cases})
    if args.phase == "research":
        return
    results = (optimize(spec, args.output, cases, research) if args.phase in ("optimize", "all") else
        [read(args.output / "runs" / f"{case['case_id']}_{arm}.json") for case in cases for arm in ARMS])
    if args.phase == "optimize":
        return
    evaluations = evaluate(spec, args.output, cases, results)
    summarize(spec, args.output, cases, research, results, evaluations)


if __name__ == "__main__":
    main()
