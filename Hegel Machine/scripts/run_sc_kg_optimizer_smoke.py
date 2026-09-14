#!/usr/bin/env python3
"""Exercise the original SC-KG acquisition loop at equal total simulator cost."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.sc_kg_environment import SCKGEnvironment, compile_sc_kg_program


class CountedProblem:
    def __init__(self, base):
        self.base = base
        self.simulator_calls = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def simulate(self, point, rng=None):
        result = self.base.simulate(point, rng)
        self.simulator_calls += 1
        return result


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((args.source / "protocol.json").read_text())
    settings = dict(n0=8, K1=6, K2=0, posterior_pool_size=32, posterior_keep=3,
                    axis_candidate_count=0, state_candidate_count=0, n_thr=2,
                    use_problem_initial_samples=False, use_boundary_initial_samples=False,
                    use_recommendation_refinement=False, recommendation_axis_oracle=False,
                    recommendation_axis_candidate_count=0, use_state_coupling=False,
                    use_state_basis=False, lambda_coupling=0.0, variance_mode="pooled",
                    acquisition_mode="exact_mc", exact_kg_mc_samples=1,
                    exact_kg_jobs=1, eval_pool_size=32, evaluate_interval=0,
                    truth_pool_diagnostics=False, checkpoint_interval=0)
    protocol = {"source": str(args.source.resolve()), "total_calls_per_arm": 64,
                "source_selection": "cold RZDT2 task at position 1 in each of the three frozen cohorts",
                "optimizer_seed_offset": 4_000_009, "settings": settings,
                "accounting": "Learned includes each frozen program's actual Hegel research calls and receives only the remaining total budget for SC-KG. Default receives all 64 calls for SC-KG. Prior calls are already incurred in the source study, and are charged here rather than reacquired.",
                "evaluation": "The original optimizer's final truth report runs only after acquisition and final recommendation freeze. Online truth diagnostics and intermediate truth evaluation are disabled.",
                "scope": "Three-seed full acquisition path smoke on noise-free RZDT2, not independent scientific confirmation or comparison with the current paper method."}
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("existing smoke protocol differs")
    write(path, protocol)
    results = []
    for cohort in source_protocol["cohorts"]:
        source_path = args.source / f"{cohort}_cold_learning.json"
        source = json.loads(source_path.read_text())["results"][1]
        task = source["task"]
        research_cost = source["actual_costs"]["simulator_calls"]
        for arm in ("learned", "default"):
            path = args.output / f"{cohort}_{arm}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            environment = SCKGEnvironment(source_protocol["sc_kg_project_root"], **task)
            from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
            from problems.single_objective import ScalarizedProblem
            prior_calls = research_cost if arm == "learned" else 0
            adapter = (compile_sc_kg_program(environment, Program.from_dict(source["result"]["program"]))
                       if arm == "learned" else CountedProblem(ScalarizedProblem(environment._problem)))
            config = SingleOLHKGConfig(N=64 - prior_calls, seed=task["seed"] + 4_000_009, **settings)
            algorithm = SingleOLHKGAlgorithm(adapter, config)
            final = algorithm.run()
            result = {"cohort": cohort, "arm": arm, "task": task,
                      "source_learning_path": str(source_path.resolve()) if arm == "learned" else None,
                      "research_simulator_calls": prior_calls,
                      "optimizer_simulator_calls": adapter.simulator_calls,
                      "total_simulator_calls": prior_calls + adapter.simulator_calls,
                      "optimizer_reported_calls": final["n_simulations"],
                      "config": asdict(config),
                      "basis_dimensions": [model.p for model in algorithm.gpr],
                      "learned_basis_evaluations": adapter.basis.evaluations if arm == "learned" else 0,
                      "initial_points": algorithm.pre_sampling_log["samples"],
                      "actions": [{key: row.get(key) for key in (
                          "stage", "x_selected", "Y_observed", "mu_before", "decision_backend",
                          "n_candidates", "exact_kg_raw_selected", "kg_selected", "action_kind")}
                                  for row in algorithm.iteration_log],
                      "final": {key: final.get(key) for key in (
                          "x_recommended", "true_objective", "true_constraint_mean", "true_feasible",
                          "simple_regret", "n_simulations", "total_time_sec")}}
            if result["total_simulator_calls"] != 64:
                raise RuntimeError("actual simulator cost differs from the fixed total budget")
            write(path, result)
            results.append(result)
            print(json.dumps({"cohort": cohort, "arm": arm, "calls": result["total_simulator_calls"],
                              "final": result["final"]}), flush=True)
    comparisons = []
    for cohort in source_protocol["cohorts"]:
        learned = next(row for row in results if row["cohort"] == cohort and row["arm"] == "learned")
        default = next(row for row in results if row["cohort"] == cohort and row["arm"] == "default")
        paired = list(zip(learned["actions"], default["actions"]))
        comparisons.append({"cohort": cohort,
            "same_initial_points": learned["initial_points"] == default["initial_points"],
            "paired_adaptive_steps": len(paired),
            "different_action_steps": sum(a["x_selected"] != b["x_selected"] for a, b in paired),
            "first_different_stage": next((a["stage"] for a, b in paired
                                            if a["x_selected"] != b["x_selected"]), None),
            "learned_minus_default_objective": learned["final"]["true_objective"] - default["final"]["true_objective"],
            "learned_feasible": learned["final"]["true_feasible"],
            "default_feasible": default["final"]["true_feasible"]})
    summary = {"protocol": protocol, "results": results, "comparisons": comparisons,
               "aggregate": {arm: {"feasible": sum(row["final"]["true_feasible"] for row in results if row["arm"] == arm),
                                   "mean_objective": mean(row["final"]["true_objective"] for row in results if row["arm"] == arm)}
                             for arm in ("learned", "default")}}
    write(args.output / "summary.json", summary)
    print(json.dumps({"aggregate": summary["aggregate"], "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
