#!/usr/bin/env python3
"""Compare data reuse and learned features inside the unchanged SC-KG search."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.sc_kg_environment import (
    HegelProgramProblem, SCKGEnvironment, compile_sc_kg_program, initialize_sc_kg_optimizer,
)
from run_sc_kg_optimizer_smoke import CountedProblem, write


class DefaultBasisResearchProblem(HegelProgramProblem):
    """Consume identical paid data with SC-KG's ordinary raw/quadratic basis."""

    def gpr_basis_map(self, output_index=1):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--prior-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_spec = json.loads((args.source_study / "protocol.json").read_text())
    old = json.loads((args.prior_smoke / "summary.json").read_text())
    arms = ("learned_with_data", "default_with_data", "default_cold")
    protocol = {"source_protocol": source_spec, "prior_smoke": str(args.prior_smoke.resolve()),
                "arms": list(arms), "total_calls_per_arm": 64,
                "optimizer_settings": old["protocol"]["settings"],
                "optimizer_seed_offset": old["protocol"]["optimizer_seed_offset"],
                "task_selection": "RZDT2 at position 1 of each source cohort; acquire new full vector records under the same frozen research protocol",
                "accounting": "Each data-reuse arm pays for the whole Hegel research history once, initializes the real SC-KG GP/HVD with every recorded objective/constraint observation, and spends the remaining calls on original SC-KG acquisition. The default-cold arm gets 64 original SC-KG calls. The two data-reuse arms share identical acquired research data; their per-arm totals each include that source cost.",
                "evaluation": "Original SC-KG post-run truth scoring only, after all acquisition and the final recommendation have frozen. No truth values reconstruct missing research outputs.",
                "scope": "Three-seed bounded optimizer comparison; the prior program-only negative baseline is retained."}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("the saved data-reuse protocol differs")
    write(protocol_path, protocol)

    research = {}
    for cohort, tasks in source_spec["cohorts"].items():
        path = args.output / f"{cohort}_research.json"
        task = tasks[1]
        environment = SCKGEnvironment(source_spec["sc_kg_project_root"], **task)
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved and saved["result"]["status"] == "completed":
            research[cohort] = saved
            continue
        library = KnowledgeLibrary.from_dict(saved["library"]) if saved else KnowledgeLibrary()
        config = replace(ResearchConfig(**source_spec["config"]), seed=task["seed"])
        def checkpoint(report):
            write(path, {"task": task, "result": report, "library": library.to_dict()})
        report = MetaPriorResearch(config, library=library).run(environment,
            research_state=saved["result"]["research_state"] if saved else None,
            on_checkpoint=checkpoint)
        checkpoint(report)
        research[cohort] = json.loads(path.read_text())
        print(json.dumps({"cohort": cohort, "research": report["status"], "costs": report["costs"]}), flush=True)

    results = []
    for cohort, source in research.items():
        task, report = source["task"], source["result"]
        for arm in arms:
            path = args.output / f"{cohort}_{arm}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            environment = SCKGEnvironment(source_spec["sc_kg_project_root"], **task)
            from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
            from problems.single_objective import ScalarizedProblem
            config = SingleOLHKGConfig(N=64,
                seed=task["seed"] + protocol["optimizer_seed_offset"], **protocol["optimizer_settings"])
            if arm == "default_cold":
                adapter = CountedProblem(ScalarizedProblem(environment._problem))
                algorithm = SingleOLHKGAlgorithm(adapter, config)
                research_calls = 0
            else:
                environment.restore(report["research_state"]["environment_state"])
                program = Program.from_dict(report["program"])
                adapter = (compile_sc_kg_program(environment, program) if arm == "learned_with_data"
                           else DefaultBasisResearchProblem(environment, program))
                algorithm = initialize_sc_kg_optimizer(adapter, config)
                research_calls = adapter.research_simulator_calls
            final = algorithm.run()
            initial_outputs = [[float(value) for value in output]
                               for _, output in algorithm.history[:algorithm.config.n0]]
            result = {"cohort": cohort, "arm": arm, "task": task,
                "research_simulator_calls": research_calls,
                "optimizer_new_simulator_calls": adapter.simulator_calls,
                "total_simulator_calls": research_calls + adapter.simulator_calls,
                "loaded_observation_count": getattr(adapter, "loaded_observation_count", 0),
                "optimizer_history_count": len(algorithm.history), "config": asdict(algorithm.config),
                "basis_dimensions": [model.p for model in algorithm.gpr],
                "learned_basis_evaluations": adapter.basis.evaluations if arm != "default_cold" else 0,
                "initial_points": algorithm.pre_sampling_log["samples"],
                "initial_outputs": initial_outputs,
                "actions": [{key: row.get(key) for key in (
                    "stage", "x_selected", "Y_observed", "mu_before", "n_candidates",
                    "exact_kg_raw_selected", "action_kind")} for row in algorithm.iteration_log],
                "final": {key: final.get(key) for key in (
                    "x_recommended", "true_objective", "true_constraint_mean", "true_feasible",
                    "simple_regret", "n_simulations", "total_time_sec")}}
            if result["total_simulator_calls"] != 64 or result["optimizer_history_count"] != 64:
                raise RuntimeError("the complete paid history and fixed total budget must both be 64")
            if arm != "default_cold" and result["loaded_observation_count"] != research_calls:
                raise RuntimeError("the optimizer did not consume every paid research observation")
            write(path, result)
            results.append(result)
            print(json.dumps({"cohort": cohort, "arm": arm, "calls": result["total_simulator_calls"],
                              "loaded": result["loaded_observation_count"], "final": result["final"]}), flush=True)
    comparisons = []
    for cohort in research:
        rows = {row["arm"]: row for row in results if row["cohort"] == cohort}
        learned, reused, cold = (rows[arm] for arm in arms)
        before = next(row for row in old["results"] if row["cohort"] == cohort and row["arm"] == "learned")
        comparisons.append({"cohort": cohort,
            "data_arms_same_initial_points": learned["initial_points"] == reused["initial_points"],
            "data_arms_same_initial_outputs": learned["initial_outputs"] == reused["initial_outputs"],
            "different_data_arm_actions": sum(a["x_selected"] != b["x_selected"]
                for a, b in zip(learned["actions"], reused["actions"])),
            "learned_with_data_minus_default_cold": learned["final"]["true_objective"] - cold["final"]["true_objective"],
            "learned_with_data_minus_default_with_data": learned["final"]["true_objective"] - reused["final"]["true_objective"],
            "learned_with_data_minus_prior_program_only": learned["final"]["true_objective"] - before["final"]["true_objective"]})
    summary = {"protocol": protocol, "results": results, "comparisons": comparisons,
        "aggregate": {arm: {"feasible": sum(row["final"]["true_feasible"] for row in results if row["arm"] == arm),
                            "mean_objective": mean(row["final"]["true_objective"] for row in results if row["arm"] == arm)} for arm in arms},
        "actual_new_research_calls": sum(item["result"]["costs"]["observation_cost"] for item in research.values()),
        "actual_new_optimizer_calls": sum(row["optimizer_new_simulator_calls"] for row in results)}
    write(args.output / "summary.json", summary)
    print(json.dumps({"aggregate": summary["aggregate"], "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
