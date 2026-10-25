#!/usr/bin/env python3
"""Compare persistent-state and risk research on the common unknown-task stream."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.evaluation import point_loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


SEEDS = (2949019033, 2949119059, 2949219071)
TASK_ORDER = (0, 2, 8, 9, 10, 11, 12, 13, 10, 8, 11, 9, 13, 12, 2, 0)
ARMS = {
    "default": {},
    "no_memory": {"use_memory": False},
    "no_state": {"state_search": False},
    "no_risk": {"risk_learning": False},
    "fixed_library": {"strategy": "fixed", "use_memory": False},
}
SOURCES = ("unknown_task_lab.py", "unified_research.py", "unified_session.py",
           "meta_prior_claims.py", "unified_uncertainty.py", "continuous_state.py",
           "risk_search.py", "research_graph.py", "language.py", "memory.py")


def write(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=False, allow_nan=True)


def tasks_for(seed):
    return [{"task_index": task_index, "seed": seed + 10007 * position,
             "profile": "development" if position < 8 else "confirmation"}
            for position, task_index in enumerate(TASK_ORDER)]


def prepare(output):
    config = ResearchConfig(observation_budget=384, compute_budget=240000,
        minimum_improvement=1e-7, max_rounds=4, active_probes=False)
    protocol = {"experiment": "unified_state_and_risk_research_v060",
        "seeds": list(SEEDS), "arms": ARMS,
        "tasks_by_seed": {str(seed): tasks_for(seed) for seed in SEEDS},
        "research_config": asdict(config), "measurement_budget": 1536,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "loss_contract": "Use the public TaskSpec objective: squared error at q=0.5, otherwise "
            "pinball loss at the advertised quantile. Summaries keep these loss kinds separate.",
        "oracle_contract": "Only the terminal evaluator obtains oracle values, alongside the same "
            "evaluation observations, after the whole learning sequence. The oracle conditions on "
            "the full noiseless physical state, including unpurchased instruments. Its realized loss "
            "separates irreducible observation noise from model/view error; finite-sample excess "
            "loss is reported without clipping. Oracle targets never enter the researcher or memory.",
        "endpoints": "Task loss, oracle loss and excess, oracle-prediction squared error, empirical "
            "quantile coverage, actual selected state/risk representations, confirmed source calls, "
            "rejected transfer, and observation/measurement/search costs. Only deterministic MSE "
            "tasks use the existing solved threshold 1e-6; risk losses are reported numerically.",
        "task_stream": "The first eight tasks use development parameters. The later eight use "
            "confirmation parameters, with recurring control/instrument renamings and decay changes.",
        "scope": "Five algorithmic ablations on this new task stream. No additional Code Agent "
            "model calls, and no reuse of old Code Agent scores as evidence on these tasks.",
        "resume": "Checkpoint every completed learning task and terminal evaluation. Resume the "
            "saved sequence and its frozen source; completed tasks and scores are not repeated.",
        "planned_learning_tasks": len(SEEDS)*len(ARMS)*len(TASK_ORDER),
        "maximum_learning_observations": len(SEEDS)*len(ARMS)*len(TASK_ORDER)*384,
        "maximum_learning_measurement_cost": len(SEEDS)*len(ARMS)*len(TASK_ORDER)*1536,
        "planned_evaluation_observations": len(SEEDS)*len(ARMS)*len(TASK_ORDER)*64*4}
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("saved study protocol differs; retain the existing experiment")
    else:
        write(path, protocol)
    return protocol


def freeze(output):
    for name in SOURCES:
        source = ROOT / "src/hegel_machine/cognition" / name
        target = output / "method_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"method changed since first execution: {name}")
        else:
            shutil.copyfile(source, target)
    target = output / "method_source" / Path(__file__).name
    source = Path(__file__).resolve()
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError("study driver changed since first execution")
    if not target.exists():
        shutil.copyfile(source, target)


def evaluate(session, settings, on_checkpoint):
    """Generate one terminal stream per task and score its fixed program."""
    if len(session.state["results"]) != len(session.state["tasks"]):
        raise ValueError("complete the learning sequence before oracle evaluation")
    for position in range(len(session.state["evaluation"]), len(session.state["tasks"])):
        item = session.state["tasks"][position]
        result = session.state["results"][position]["research"]
        with session.lab_factory(item) as laboratory:
            reference = laboratory.evaluation_oracle(result["selected_channels"],
                seed=item["seed"] + settings["seed_offset"], length=settings["length"],
                trajectories=settings["trajectories"])
        rows = reference["observations"]
        quantile = reference["risk_quantile"]
        program = Program.from_dict(result["program"])
        predictions = [predict(program, row, result["fallback"]) for row in rows]
        optimal = reference["conditional_means"] if quantile == .5 else reference["conditional_quantiles"]
        measured_loss = mean(point_loss(value, row.target, quantile) for value, row in zip(predictions, rows))
        oracle_loss = mean(point_loss(value, row.target, quantile) for value, row in zip(optimal, rows))
        score = {"task_position": position, "rows": len(rows),
            "loss_kind": "mse" if quantile == .5 else "pinball", "risk_quantile": quantile,
            "loss": measured_loss, "oracle_loss": oracle_loss, "excess_loss": measured_loss-oracle_loss,
            "oracle_prediction_mse": mean((value-truth)**2 for value, truth in zip(predictions, optimal)),
            "empirical_coverage": mean(row.target <= value for row, value in zip(rows, predictions)),
            "oracle_empirical_coverage": mean(row.target <= value for row, value in zip(rows, optimal)),
            "observation_cost": reference["costs"]["observation_cost"],
            "measurement_cost": reference["costs"]["measurement_cost"], "compute_cost": len(rows)}
        if quantile == .5:
            score["mse"] = measured_loss
        session.state["evaluation"].append(score)
        on_checkpoint(session.to_dict())
    return session.to_dict()


def research(output, protocol, *, selected_arm=None, selected_seed=None, max_tasks=None):
    freeze(output)
    for seed in protocol["seeds"]:
        if selected_seed is not None and seed != selected_seed:
            continue
        for arm, overrides in protocol["arms"].items():
            if selected_arm is not None and arm != selected_arm:
                continue
            path = output / "runs" / f"{seed}_{arm}.json"
            saved = json.loads(path.read_text()) if path.exists() else None
            tasks = protocol["tasks_by_seed"][str(seed)]
            if saved and len(saved["evaluation"]) == len(tasks):
                continue
            config = ResearchConfig(**{**protocol["research_config"], "seed": seed, **overrides})
            session = UnifiedResearchSession(config, tasks=tasks, state=saved,
                measurement_budget=protocol["measurement_budget"])

            def checkpoint(report):
                write(path, report)

            start = len(session.state["results"])
            session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
            print(json.dumps({"seed": seed, "arm": arm, "completed_tasks": len(session.state["results"]),
                "new_tasks": len(session.state["results"])-start,
                "costs": session.to_dict()["costs"]}), flush=True)
            if len(session.state["results"]) == len(tasks):
                report = evaluate(session, protocol["evaluation"], checkpoint)
                for kind in ("mse", "pinball"):
                    selected = [row for row in report["evaluation"] if row["loss_kind"] == kind]
                    print(json.dumps({"seed": seed, "arm": arm, "loss_kind": kind,
                        "tasks": len(selected), "mean_loss": mean(row["loss"] for row in selected),
                        "oracle_loss": mean(row["oracle_loss"] for row in selected),
                        "excess_loss": mean(row["excess_loss"] for row in selected)}), flush=True)


def program_details(program):
    nodes = []

    def visit(node):
        nodes.append(node)
        for child in node.children:
            visit(child)

    visit(program.expanded())
    return {"operators": dict(Counter(node.op for node in nodes)),
            "accumulate_decays": sorted({node.value for node in nodes if node.op == "accumulate"})}


def summarize(output, protocol):
    runs = []
    for path in sorted((output / "runs").glob("*.json")):
        seed, arm = path.stem.split("_", 1)
        report = json.loads(path.read_text())
        scores = {row["task_position"]: row for row in report["evaluation"]}
        tasks = []
        for position, result in enumerate(report["results"]):
            research_result = result["research"]
            tasks.append({"task_position": position, **report["tasks"][position],
                "status": research_result["status"], "selected_kind": research_result["selected_kind"],
                "selected_channels": research_result["selected_channels"],
                **program_details(Program.from_dict(research_result["program"])),
                "source_calls": research_result["used_sources"],
                "method_source_ids": research_result.get("method_source_ids", []),
                "rejected_transfers": sum(event["status"] == "counterexample"
                    for event in research_result["transfer"]),
                "costs": research_result["costs"], "evaluation": scores.get(position)})
        runs.append({"seed": int(seed), "arm": arm, "status": report["status"], "tasks": tasks,
            "costs": report["costs"], "evaluation_costs": {
                key: sum(row[key] for row in report["evaluation"])
                for key in ("observation_cost", "measurement_cost", "compute_cost")}})

    def metrics(tasks):
        evaluated = [task for task in tasks if task["evaluation"] is not None]
        by_loss = {}
        for kind in ("mse", "pinball"):
            rows = [task["evaluation"] for task in evaluated if task["evaluation"]["loss_kind"] == kind]
            if rows:
                by_loss[kind] = {"tasks": len(rows), **{key: mean(row[key] for row in rows)
                    for key in ("loss", "oracle_loss", "excess_loss", "oracle_prediction_mse",
                                "empirical_coverage", "oracle_empirical_coverage")}}
                if kind == "mse":
                    by_loss[kind]["solved"] = sum(row["loss"] <= 1e-6 for row in rows)
        return {"tasks": len(tasks), "evaluated_tasks": len(evaluated), "losses": by_loss,
            "research_statuses": dict(Counter(task["status"] for task in tasks)),
            "selected_kinds": dict(Counter(task["selected_kind"] for task in tasks)),
            "accumulate_tasks": sum(bool(task["accumulate_decays"]) for task in tasks),
            "confirmed_source_uses": sum(bool(task["source_calls"]) for task in tasks),
            "rejected_transfers": sum(task["rejected_transfers"] for task in tasks),
            "learning_costs": {key: sum(task["costs"][key] for task in tasks)
                               for key in ("observation_cost", "measurement_cost", "compute_cost")}}

    arms = {}
    for arm in protocol["arms"]:
        tasks = [task for run in runs if run["arm"] == arm for task in run["tasks"]]
        arms[arm] = {"all_tasks": metrics(tasks),
                     "changed_parameter_successors": metrics([task for task in tasks if task["task_position"] >= 8])}
    summary = {"experiment": protocol["experiment"], "arms": arms, "runs": runs,
        "completed_sequences": sum(len(run["tasks"]) == len(TASK_ORDER)
            and all(task["evaluation"] is not None for task in run["tasks"]) for run in runs),
        "actual_learning_observations": sum(run["costs"]["observation_cost"] for run in runs),
        "actual_learning_measurement_cost": sum(run["costs"]["measurement_cost"] for run in runs),
        "actual_evaluation_observations": sum(run["evaluation_costs"]["observation_cost"] for run in runs),
        "actual_evaluation_measurement_cost": sum(run["evaluation_costs"]["measurement_cost"] for run in runs)}
    write(output / "summary.json", summary)
    print(json.dumps({"completed_sequences": summary["completed_sequences"], "arms": arms}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "research", "summary"), default="prepare")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--max-tasks", type=int)
    arguments = parser.parse_args()
    protocol = prepare(arguments.output)
    if arguments.phase == "research":
        research(arguments.output, protocol, selected_arm=arguments.arm,
                 selected_seed=arguments.seed, max_tasks=arguments.max_tasks)
    elif arguments.phase == "summary":
        summarize(arguments.output, protocol)
    else:
        print("Prepared three seeds, five algorithmic arms and 16 arriving tasks per sequence.")
