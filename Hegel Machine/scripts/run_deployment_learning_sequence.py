#!/usr/bin/env python3
"""Measure whether deployment discoveries help the next research task."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition import sc_kg_environment as adapter
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.meta_prior_session import MetaPriorSession
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig
from run_sc_kg_prior_mechanism_change import MechanismChangeProblem

SEEDS = (2647017009, 2647117031, 2647217053)
ARMS = ("batched_v057", "interleaved", "no_deployment_feedback")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(output):
    protocol = {
        "experiment": "deployment_discovery_to_successor_learning",
        "seeds": list(SEEDS), "arms": list(ARMS),
        "problems": ["state_policy_rzdt1", "rzdt2", "state_policy_rzdt1", "rzdt2"],
        "source": {"kind": "sc_kg", "project_root": "/home/erzhu419/mine_code/KG_op/SC-OLH-KG",
                   "dimension": 3, "noise": 0.0},
        "research_config": asdict(ResearchConfig(observation_budget=32, compute_budget=60000,
            minimum_improvement=1e-7, max_rounds=2)),
        "optimizer_config": json.loads((ROOT / "config/cognition_sc_kg_optimizer.json").read_text()),
        "first_task_change_after_optimizer_calls": 12, "evaluation_rows_per_task": 128,
        "baseline_source": str((output / "baseline_source/meta_prior_session.py").resolve()),
        "method": "Keep0.57 research and optimizer policies fixed. Change only session ordering: "
            "research then deploy and retain revisions before starting the next task. The batched "
            "control researches all tasks before any deployment. The no-feedback control interleaves "
            "tasks but suppresses deployment revision retention while retaining initial research memory.",
        "task_information": "All researchers see the same public controls and scalar measurements. "
            "Only the first task changes its actual simulator after12 optimizer observations. Its "
            "learned revision can help two subsequent related tasks, with an intervening original-mechanism task.",
        "endpoints": "Successor research observations and compute, independent prediction loss, "
            "actual source-concept calls and rejected transfers, final optimizer objective/feasibility. "
            "The fixed total budget is 4 times 64 simulator calls per arm and sequence; fewer initial "
            "research observations leave more optimizer observations, not a claim of fewer total calls. "
            "Prediction evaluation scores initial research programs in each task's initial world; "
            "optimizer recommendations are scored in its final world. Primary prediction endpoints "
            "use successor tasks, whose worlds are stationary.",
        "planned_optimizer_runs": 36, "planned_learning_calls": 2304,
        "planned_evaluation_calls": 4608, "planned_physical_calls": 6912,
        "limitations": "Development experiment on one four-task local simulator sequence. "
            "It tests the deployment feedback connection, not general autonomous research routing, "
            "a new structure family, or superiority over a general code Agent. No code-model calls."}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("saved sequence protocol differs")
    if not path.exists():
        write(path, protocol)
    return protocol


def baseline_class(path):
    name = "hegel_machine.cognition._sequence_v057"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MetaPriorSession


@contextmanager
def changing_first_task(seed):
    original = adapter.compile_sc_kg_program

    def compile_program(environment, program):
        if environment._settings["seed"] == seed:
            return MechanismChangeProblem(environment, program, change_after=12)
        return original(environment, program)

    adapter.compile_sc_kg_program = compile_program
    try:
        yield
    finally:
        adapter.compile_sc_kg_program = original


class NoDeploymentFeedback(MetaPriorSession):
    def _retain_optimizer_revisions(self, spec, recovery, source_ids):
        return []

    def _record_optimizer_trials(self, research, result):
        return []


def run(output, protocol):
    baseline = baseline_class(protocol["baseline_source"])
    for seed in protocol["seeds"]:
        for arm in ARMS:
            path = output / "runs" / f"{seed}_{arm}.json"
            if path.exists():
                continue
            cls = baseline if arm == ARMS[0] else NoDeploymentFeedback if arm == ARMS[2] else MetaPriorSession
            checkpoint = output / "checkpoints" / f"{seed}_{arm}.json"
            session = (cls(state=json.loads(checkpoint.read_text())) if checkpoint.exists() else
                cls(config=ResearchConfig(**{**protocol["research_config"], "seed": seed}),
                    source={**protocol["source"], "seed": seed, "problems": protocol["problems"]}))
            with changing_first_task(seed):
                if arm == ARMS[0]:
                    session.run(on_checkpoint=lambda state: write(checkpoint, state))
                    session.optimize(protocol["optimizer_config"],
                        on_checkpoint=lambda state: write(checkpoint, state))
                else:
                    session.run_deploy(protocol["optimizer_config"],
                        on_checkpoint=lambda state: write(checkpoint, state))
            report = session.evaluate(count=protocol["evaluation_rows_per_task"])
            report.update(seed=seed, arm=arm)
            write(path, report)
            print(json.dumps({"seed": seed, "arm": arm, "research": report["costs"],
                "revisions": sum(len(r["revision_prior_updates"]) for r in report["optimizations"]),
                "mse": mean(row["mse"] for row in report["evaluation"])}), flush=True)
    summarize(output)


def summarize(output):
    sequences = []
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        deployment_ids = {item["source_concept_id"] for row in report["optimizations"]
                          for item in row["revision_prior_updates"]}
        deployment_prior_ids = {item["prior_id"] for row in report["optimizations"]
                                for item in row["revision_prior_updates"]}
        tasks = []
        for row, optimization, evaluation in zip(report["results"], report["optimizations"], report["evaluation"]):
            research = row["research"]
            sources = _sources(Program.from_dict(research["program"])) if research["program"] else []
            available_priors = [prior for prior in report["knowledge"]["meta_priors"]
                               if prior["prior_id"] in row.get("available_prior_ids", [])]
            available_deployment_ids = {prior["source_concept_id"] for prior in available_priors
                                       if prior["prior_id"] in deployment_prior_ids}
            lineage, pending = set(sources), list(sources)
            while pending:
                concept_id = pending.pop()
                for prior in available_priors:
                    if prior["source_concept_id"] == concept_id:
                        for parent in prior["parent_concept_ids"]:
                            if parent not in lineage:
                                lineage.add(parent)
                                pending.append(parent)
            tasks.append({"position": row["position"], "research_costs": research["costs"],
                "source_concepts": sources,
                "available_deployment_concepts": sorted(available_deployment_ids),
                "deployment_source_calls": sorted(available_deployment_ids.intersection(sources)),
                "deployment_source_ancestry": sorted(available_deployment_ids.intersection(lineage)),
                "counterexamples": len(research["research_state"]["counterexamples"]),
                "mse": evaluation["mse"], "optimizer_calls": optimization["new_optimizer_calls"],
                "total_learning_calls": optimization["total_simulator_calls"],
                "objective_compute": optimization["objective_research"]["compute_cost"],
                "revision_compute": optimization["hypothesis_revision"]["costs"]["compute_cost"],
                "final": optimization["final"]})
        sequences.append({"seed": report["seed"], "arm": report["arm"], "tasks": tasks,
            "deployment_concepts": sorted(deployment_ids), "record": str(path.resolve())})
    metrics = {}
    for arm in ARMS:
        tasks = [task for row in sequences if row["arm"] == arm for task in row["tasks"]]
        future = [task for task in tasks if task["position"] > 0]
        if not tasks:
            continue
        metrics[arm] = {"tasks": len(tasks), "successor_tasks": len(future),
            "successor_research_observations": sum(t["research_costs"]["observation_cost"] for t in future),
            "successor_research_compute": sum(t["research_costs"]["compute_cost"] for t in future),
            "successor_mse": mean(t["mse"] for t in future),
            "successor_deployment_source_calls": sum(bool(t["deployment_source_calls"]) for t in future),
            "total_learning_calls": sum(t["total_learning_calls"] for t in tasks),
            "objective_compute": sum(t["objective_compute"] for t in tasks),
            "revision_compute": sum(t["revision_compute"] for t in tasks),
            "feasible": sum(t["final"]["true_feasible"] for t in tasks),
            "final_objective": mean(t["final"]["true_objective"] for t in tasks)}
    learning = sum(value["total_learning_calls"] for value in metrics.values())
    protocol = json.loads((output / "protocol.json").read_text())
    evaluated = sum(len(row["tasks"]) for row in sequences) * protocol["evaluation_rows_per_task"]
    write(output / "summary.json", {"sequences": sequences, "metrics": metrics,
        "actual_optimizer_runs": sum(len(row["tasks"]) for row in sequences),
        "learning_calls": learning, "evaluation_calls": evaluated, "physical_calls": learning + evaluated})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run", "summary"), default="plan")
    args = parser.parse_args()
    protocol = prepare(args.output)
    if args.phase == "run":
        run(args.output, protocol)
    elif args.phase == "summary":
        summarize(args.output)
    else:
        print(json.dumps({"status": "prepared", "physical_calls": protocol["planned_physical_calls"]}))
