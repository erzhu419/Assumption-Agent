#!/usr/bin/env python3
"""Compare measured prior adoption with fixed learned and default SC-KG bases."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.sc_kg_environment import (
    HegelProgramProblem, SCKGEnvironment, compile_sc_kg_program, initialize_sc_kg_optimizer,
)


DEVELOPMENT_SEEDS = (843010020, 843110026, 843210036)
FORWARD_SEEDS = (943010033, 943110047, 943210059)
NOISE_LEVELS = (0.0, 0.03)
ARMS = ("auto", "learned", "default")


class DefaultBasisProblem(HegelProgramProblem):
    def gpr_basis_map(self, output_index=1):
        return None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, body):
    write_json_checkpoint(path, body, indent=2, ensure_ascii=False, allow_nan=True)


def protocol(source_study):
    source = read(source_study / "protocol.json")
    config = replace(ResearchConfig(**source["source_protocol"]["config"]),
                     observation_budget=32, max_rounds=2)
    cases = []
    for split, seeds in (("development", DEVELOPMENT_SEEDS), ("forward", FORWARD_SEEDS)):
        for index, seed in enumerate(seeds, 1):
            for noise in NOISE_LEVELS:
                case = {"case_id": f"{split}_{index}_noise_{noise:.2f}", "split": split, "profile": "rzdt2",
                    "task": {"problem": "rzdt2", "dimension": 5, "seed": seed, "noise": noise}}
                if split == "development" and noise == 0.:
                    case["existing_research"] = str((source_study / f"sequence_{index}_research.json").resolve())
                cases.append(case)
    for index, seed in enumerate(FORWARD_SEEDS, 1):
        cases.append({"case_id": f"structure_{index}_noise_0.00", "split": "forward",
            "profile": "state_policy_rzdt1",
            "task": {"problem": "state_policy_rzdt1", "dimension": 3, "seed": seed, "noise": 0.}})
    return {"experiment": "prospective_sc_kg_prior_adoption", "source_study": str(source_study.resolve()),
        "sc_kg_project_root": source["source_protocol"]["sc_kg_project_root"],
        "cases": cases, "arms": list(ARMS), "total_calls_per_arm": 64,
        "research_config": asdict(config), "optimizer_settings": source["optimizer_settings"],
        "optimizer_seed_offset": source["optimizer_seed_offset"],
        "policy": {"initial_basis": "default", "minimum_new_observations": 4, "minimum_gain": 1e-7,
            "evidence": "Paired constraint predictions made before each new optimizer observation; the research data fit both models but do not count as prior-adoption trials"},
        "accounting": "Every arm loads exactly the same actual objective/constraint research vectors and includes those paid calls in its total of 64. Adoption uses the optimizer's ordinary new observations and adds no simulator calls. Reused historical research and newly acquired research are recorded separately.",
        "candidate_status": "An unresolved noisy research run supplies only its actual incumbent candidate; it is not marked as a confirmed world claim.",
        "evaluation": "All research is completed before optimization. Each original SC-KG run freezes its recommendation before truth scoring; evaluate_interval=0 and truth_pool_diagnostics=false. Truth values never enter adoption or reconstruct an unrecorded output.",
        "scope": "RZDT2: three previously observed development seeds and three frozen new seeds, each at noise 0 and 0.03. Separately, StatePolicyRZDT1 dimension three at the three new seeds and noise zero tests a genuine nonquadratic response boundary. Profiles, splits and noise levels are reported separately, including failed adoption and withdrawal."}


def acquire_research(spec, output):
    reports = {}
    for case in spec["cases"]:
        path = output / f"{case['case_id']}_research.json"
        if path.exists():
            saved = read(path)
            if saved.get("finished"):
                reports[case["case_id"]] = saved
                continue
        else:
            saved = None
        if case.get("existing_research"):
            source = read(case["existing_research"])
            if source["task"] != case["task"]:
                raise ValueError("the reused research does not match the frozen case")
            saved = {**source, "finished": True, "case_id": case["case_id"],
                "research_origin": case["existing_research"], "actual_new_research_calls": 0}
            write(path, saved)
            reports[case["case_id"]] = saved
            continue
        environment = SCKGEnvironment(spec["sc_kg_project_root"], **case["task"])
        library = KnowledgeLibrary.from_dict(saved["library"]) if saved else KnowledgeLibrary()
        config = replace(ResearchConfig(**spec["research_config"]), seed=case["task"]["seed"])

        def checkpoint(report):
            write(path, {"case_id": case["case_id"], "task": case["task"], "result": report,
                "library": library.to_dict(), "research_origin": "new_public_measurements",
                "actual_new_research_calls": report["costs"]["observation_cost"], "finished": False})

        report = MetaPriorResearch(config, library).run(environment,
            research_state=saved["result"]["research_state"] if saved else None, on_checkpoint=checkpoint)
        checkpoint(report)
        saved = read(path)
        saved["finished"] = True
        write(path, saved)
        reports[case["case_id"]] = saved
        print(json.dumps({"case": case["case_id"], "research_status": report["status"],
                          "research_costs": report["costs"]}), flush=True)
    return reports


def optimize(spec, output, research):
    # These optional integration modules are imported only when executing the
    # frozen study; planning and collecting research do not start optimization.
    from hegel_machine.cognition.prior_adoption import PriorAdoption

    results = []
    for case in spec["cases"]:
        source = research[case["case_id"]]
        report = source["result"]
        payload = report.get("program") or (report["research_state"].get("incumbent") or {}).get("candidate", {}).get("program")
        if payload is None:
            raise ValueError("research did not produce an executable incumbent candidate")
        for arm in ARMS:
            path = output / f"{case['case_id']}_{arm}.json"
            if path.exists():
                results.append(read(path))
                continue
            environment = SCKGEnvironment(spec["sc_kg_project_root"], **case["task"])
            environment.restore(report["research_state"]["environment_state"])
            from algorithms.single_olhkg import SingleOLHKGConfig
            config = SingleOLHKGConfig(N=spec["total_calls_per_arm"],
                seed=case["task"]["seed"]+spec["optimizer_seed_offset"], **spec["optimizer_settings"])
            program = Program.from_dict(payload)
            adapter = (DefaultBasisProblem(environment, program) if arm == "default"
                       else compile_sc_kg_program(environment, program))
            adoption = (PriorAdoption(minimum_gain=spec["policy"]["minimum_gain"],
                                      batch_size=spec["policy"]["minimum_new_observations"])
                        if arm == "auto" else None)
            algorithm = (initialize_sc_kg_optimizer(adapter, config, adoption_policy=adoption)
                         if adoption is not None else initialize_sc_kg_optimizer(adapter, config))
            final = algorithm.run()
            result = {"case_id": case["case_id"], "split": case["split"], "profile": case["profile"], "arm": arm,
                "task": case["task"], "research_status": report["status"],
                "prior_family": report.get("family_id"), "prior_kind": report.get("kind"),
                "candidate_program": payload, "research_simulator_calls": adapter.research_simulator_calls,
                "optimizer_new_simulator_calls": adapter.simulator_calls,
                "total_simulator_calls": adapter.research_simulator_calls+adapter.simulator_calls,
                "loaded_observation_count": adapter.loaded_observation_count,
                "optimizer_history_count": len(algorithm.history), "config": asdict(algorithm.config),
                "initial_points": algorithm.pre_sampling_log["samples"],
                "initial_outputs": [[float(value) for value in vector]
                                    for _, vector in algorithm.history[:algorithm.config.n0]],
                "actions": [{key: row.get(key) for key in ("stage", "x_selected", "Y_observed",
                    "mu_before", "exact_kg_raw_selected", "action_kind")} for row in algorithm.iteration_log],
                "basis_dimensions": [model.p for model in algorithm.gpr],
                "learned_basis_evaluations": adapter.basis.evaluations,
                "adoption_policy": adoption.to_dict() if adoption is not None else None,
                "adoption_events": algorithm.adoption_events if adoption is not None else [],
                "adoption_diagnostics": algorithm.adoption_diagnostics if adoption is not None else {},
                "final_constraint_basis": algorithm.active_constraint_basis if adoption is not None else arm,
                "final": {key: final.get(key) for key in ("x_recommended", "true_objective",
                    "true_constraint_mean", "true_feasible", "simple_regret", "n_simulations", "total_time_sec")}}
            if result["total_simulator_calls"] != spec["total_calls_per_arm"] or result["optimizer_history_count"] != spec["total_calls_per_arm"]:
                raise RuntimeError("the complete paid history must equal the frozen total call budget")
            if result["loaded_observation_count"] != adapter.research_simulator_calls:
                raise RuntimeError("the optimizer failed to consume all paid research observations")
            write(path, result)
            results.append(result)
            print(json.dumps({"case": case["case_id"], "arm": arm,
                "basis": result["final_constraint_basis"], "final": result["final"]}), flush=True)
    return results


def summarize(spec, research, results):
    comparisons = []
    for case in spec["cases"]:
        selected = {row["arm"]: row for row in results if row["case_id"] == case["case_id"]}
        auto, learned, default = (selected[arm] for arm in ARMS)
        switches = [event for event in auto["adoption_events"]
                    if event.get("selected_before") != event.get("selected_after")]
        comparisons.append({"case_id": case["case_id"], "split": case["split"], "profile": case["profile"], "noise": case["task"]["noise"],
            "same_initial_points": auto["initial_points"] == learned["initial_points"] == default["initial_points"],
            "same_initial_outputs": auto["initial_outputs"] == learned["initial_outputs"] == default["initial_outputs"],
            "adoptions": sum(event.get("selected_after") == "learned" for event in switches),
            "withdrawals": sum(event.get("selected_after") == "default" for event in switches),
            "auto_minus_learned_objective": auto["final"]["true_objective"]-learned["final"]["true_objective"],
            "auto_minus_default_objective": auto["final"]["true_objective"]-default["final"]["true_objective"],
            "auto_feasible": auto["final"]["true_feasible"], "learned_feasible": learned["final"]["true_feasible"],
            "default_feasible": default["final"]["true_feasible"]})
    aggregate = []
    groups = list(dict.fromkeys((case["profile"], case["split"], case["task"]["noise"]) for case in spec["cases"]))
    for profile, split, noise in groups:
        for arm in ARMS:
            rows = [row for row in results if row["profile"] == profile and row["split"] == split
                    and row["task"]["noise"] == noise and row["arm"] == arm]
            aggregate.append({"profile": profile, "split": split, "noise": noise, "arm": arm, "tasks": len(rows),
                "feasible": sum(row["final"]["true_feasible"] for row in rows),
                "mean_objective": mean(row["final"]["true_objective"] for row in rows),
                "mean_simple_regret": mean(row["final"]["simple_regret"] for row in rows),
                "research_calls": sum(row["research_simulator_calls"] for row in rows),
                "optimizer_new_calls": sum(row["optimizer_new_simulator_calls"] for row in rows)})
    return {"protocol": spec, "comparisons": comparisons, "aggregate": aggregate, "results": results,
        "actual_new_research_calls": sum(item["actual_new_research_calls"] for item in research.values()),
        "historically_reused_research_calls": sum(item["result"]["costs"]["observation_cost"]
            for item in research.values() if item["research_origin"] != "new_public_measurements"),
        "actual_new_optimizer_calls": sum(row["optimizer_new_simulator_calls"] for row in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "optimize", "all"), default="plan")
    args = parser.parse_args()
    spec = protocol(args.source_study)
    path = args.output / "protocol.json"
    if path.exists() and read(path) != spec:
        raise ValueError("the saved prior-adoption protocol differs from this invocation")
    write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "cases": len(spec["cases"]), "arms": list(ARMS)}))
        return
    research = (acquire_research(spec, args.output) if args.phase in ("research", "all") else
                {case["case_id"]: read(args.output / f"{case['case_id']}_research.json") for case in spec["cases"]})
    if args.phase == "research":
        return
    results = optimize(spec, args.output, research)
    summary = summarize(spec, research, results)
    write(args.output / "summary.json", summary)
    print(json.dumps({"aggregate": summary["aggregate"], "comparisons": summary["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
