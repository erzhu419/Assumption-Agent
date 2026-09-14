#!/usr/bin/env python3
"""Compare research, active experiments, memory and Python on one task stream."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession, task_sequence

SEEDS = (2848019017, 2848119041, 2848219057)
ARMS = ("full", "fixed_library", "no_active", "no_memory")


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(output):
    config = ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_rounds=4)
    protocol = {"experiment": "unified_unknown_research_and_transfer", "seeds": list(SEEDS),
        "arms": [*ARMS, "ordinary_code"], "tasks_by_seed": {str(seed): task_sequence(seed) for seed in SEEDS},
        "research_config": asdict(config), "measurement_budget": 768,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "code_seed": SEEDS[0], "code_config": {"observation_budget": 192,
            "measurement_budget": 768, "compute_budget": 120000, "execution_seconds": 60,
            "max_steps": 12, "max_model_calls": 6, "token_budget": 80000},
        "cohorts": "All five arms use the first 16-task seed. Four algorithmic arms also use "
            "two further seeds. Code and Hegel share public controls, instruments, past outputs, "
            "per-task observation/measurement limits and scored-row caps. Python runtime and "
            "actual model usage are reported separately from symbolic search computation.",
        "information": "Agent receives only public controls, purchasable neutral instruments and paid "
            "observations. It is not given task_index/profile, mechanism family, relevant sensor or delay. "
            "All final predictions are scored on independent complete trajectories after sequence learning.",
        "endpoints": "Prediction MSE and solved coverage for all tasks and successors, measurement/observation/"
            "search costs, actual callable transfer, rejected transfer and selected joint revisions. "
            "All missing/failed models remain in coverage; common-complete errors are reported separately.",
        "scope": "Development comparison on eight constructed mechanisms in a 16-task stream. "
            "This tests integrated finite discovery and reuse, not all 22 meta-priors or open-world generality.",
        "planned_research_tasks": 192, "planned_code_tasks": 16,
        "maximum_learning_observations": 39936, "maximum_evaluation_observations": 53248,
        "maximum_code_calls": 96, "code_token_limit_kind": "Per-task stop between calls; last call may exceed limit."}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("saved protocol differs; retain the existing experiment")
    if not path.exists():
        write(path, protocol)
    return protocol


def freeze(output, names):
    for name in names:
        source = ROOT / "src/hegel_machine/cognition" / name
        target = output / "method_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"method changed since first execution: {name}")
        else:
            shutil.copyfile(source, target)


def research(output, protocol):
    freeze(output, ("unknown_task_lab.py", "unified_research.py", "unified_session.py"))
    for seed in protocol["seeds"]:
        tasks = protocol["tasks_by_seed"][str(seed)]
        for arm in ARMS:
            path = output / "runs" / f"{seed}_{arm}.json"
            state = json.loads(path.read_text()) if path.exists() else None
            if state and len(state["evaluation"]) == len(tasks):
                continue
            settings = {**protocol["research_config"], "seed": seed}
            if arm == "no_memory":
                settings["use_memory"] = False
            elif arm == "no_active":
                settings["active_probes"] = False
            elif arm == "fixed_library":
                settings.update(strategy="fixed", use_memory=False)
            session = UnifiedResearchSession(ResearchConfig(**settings), tasks=tasks,
                measurement_budget=protocol["measurement_budget"], state=state)
            session.run(on_checkpoint=lambda report: write(path, report))
            report = session.evaluate(**protocol["evaluation"])
            write(path, report)
            print(json.dumps({"seed": seed, "arm": arm, "costs": report["costs"],
                "mse": mean(row["mse"] for row in report["evaluation"]),
                "solved": sum(row["mse"] <= 1e-6 for row in report["evaluation"])}), flush=True)


def code(output, protocol):
    from hegel_machine.cognition.unified_code_control import UnifiedCodeSession
    freeze(output, ("unknown_task_lab.py", "unified_code_control.py", "code_agent.py"))
    seed = protocol["code_seed"]
    path = output / "runs" / f"{seed}_ordinary_code.json"
    state = json.loads(path.read_text()) if path.exists() else None
    if state and state["evaluation"] is not None:
        return
    session = UnifiedCodeSession(output / "code_workspace", tasks=protocol["tasks_by_seed"][str(seed)],
        config=protocol["code_config"], state=state)

    def checkpoint(report):
        write(path, report)
        if report["active"] is None:
            print(json.dumps({"arm": "ordinary_code", "completed_tasks": len(report["results"]),
                "status": report["status"], "costs": report["costs"]}), flush=True)

    report = session.run(on_checkpoint=checkpoint)
    if len(report["results"]) == len(report["tasks"]):
        session.evaluate(**protocol["evaluation"])
        write(path, session.to_dict())


def passive(output, protocol):
    from hegel_machine.cognition.passive_research import FixedExperimentResearchAgent
    freeze(output, ("passive_research.py",))
    plan = {"arms": ["passive_experiments"], "seeds": protocol["seeds"],
        "design": "All four 24-row random trajectories and all instruments fixed before any response. "
            "Same representation research and shared callable memory; no adaptive query, instrument or sample count.",
        "scope": "Additional requested no-active-experiment control. The earlier no_active arm only disables "
            "targeted control probes and still adapts instrument acquisition; both are retained separately."}
    write(output / "additional_controls.json", plan)
    for seed in protocol["seeds"]:
        path = output / "runs" / f"{seed}_passive_experiments.json"
        saved = json.loads(path.read_text()) if path.exists() else None
        settings = ResearchConfig(**{**protocol["research_config"], "seed": seed, "active_probes": False})
        session = UnifiedResearchSession(settings, tasks=protocol["tasks_by_seed"][str(seed)],
            measurement_budget=protocol["measurement_budget"], state=saved)
        for position in range(len(session.state["results"]), len(session.state["tasks"])):
            item = session.state["tasks"][position]
            available = [concept.concept_id for concept in session.library.concepts]
            with session.lab_factory(item) as laboratory:
                result = FixedExperimentResearchAgent(replace(settings, seed=item["seed"]),
                    measurement_budget=protocol["measurement_budget"], library=session.library).run(laboratory)
            session.state["results"].append({"task_position": position,
                "available_concepts": available, "research": result})
            session.state["status"] = "completed" if position + 1 == len(session.state["tasks"]) else "paused"
            write(path, session.to_dict())
        report = session.evaluate(**protocol["evaluation"])
        write(path, report)
        print(json.dumps({"seed": seed, "arm": "passive_experiments", "costs": report["costs"],
            "solved": sum(row["mse"] <= 1e-6 for row in report["evaluation"])}), flush=True)


def summarize(output, protocol):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
    runs = []
    for path in sorted((output / "runs").glob("*.json")):
        seed = int(path.name.split("_", 1)[0])
        arm = path.stem.split("_", 1)[1]
        report = json.loads(path.read_text())
        evaluated = report["evaluation"] or []
        evaluation_measurement_cost = 0
        for score in evaluated:
            if arm != "ordinary_code":
                evaluation_measurement_cost += score["measurement_cost"]
            elif score["rows"]:
                position = score["task_position"]
                with UnknownTaskLab(**report["tasks"][position]) as laboratory:
                    rate = sum(channel.cost_per_row for channel in laboratory.channels
                        if channel.name in report["results"][position]["selected_channels"])
                evaluation_measurement_cost += score["rows"] * rate
        rows = []
        for position, result in enumerate(report["results"]):
            score = next((row for row in evaluated if row["task_position"] == position), {})
            research_result = result if arm == "ordinary_code" else result["research"]
            if arm == "ordinary_code":
                source_ids = sorted(set(research_result["reused_files"]) &
                                    set(research_result["inherited_files"]))
                history_depth = None
            else:
                program = Program.from_dict(research_result["program"])
                source_ids = research_result["used_sources"]
                history_depth = program.history_depth
            rows.append({"position": position, "status": research_result["status"],
                "costs": research_result["costs"], "mse": score.get("mse"),
                "selected_channels": research_result["selected_channels"],
                "source_calls": source_ids, "history_depth": history_depth,
                "method_source_ids": research_result.get("method_source_ids", []),
                "declared_reused_files": research_result.get("reused_files", []),
                "source_evidence": "agent_declared_shared_files" if arm == "ordinary_code" else "confirmed_executable_call",
                "rejected_transfer_trials": sum(event["status"] == "counterexample"
                    for event in research_result.get("transfer", []))})
        runs.append({"seed": seed, "arm": arm, "status": report["status"], "tasks": rows,
            "reused_from": report.get("reused_from"),
            "evaluation_rows": sum(row["rows"] for row in evaluated),
            "evaluation_measurement_cost": evaluation_measurement_cost, "costs": report["costs"]})

    def metrics(selected):
        rows = [task for run in selected for task in run["tasks"]]
        complete = [row for row in rows if row["mse"] is not None]
        future = [row for row in rows if row["position"] > 0]
        return {"tasks": len(rows), "evaluated_models": len(complete),
            "solved": sum(row["mse"] <= 1e-6 for row in complete),
            "mean_mse_evaluated": mean(row["mse"] for row in complete) if complete else None,
            "learning_costs": {key: sum(row["costs"][key] for row in rows)
                for key in ("observation_cost", "measurement_cost", "compute_cost")},
            "successor_source_calls": sum(bool(row["source_calls"]) for row in future),
            "successor_method_uses": sum(bool(row["method_source_ids"]) for row in future),
            "successor_costs": {key: sum(row["costs"][key] for row in future)
                for key in ("observation_cost", "measurement_cost", "compute_cost")},
            "evaluation_rows": sum(run["evaluation_rows"] for run in selected),
            "evaluation_measurement_cost": sum(run["evaluation_measurement_cost"] for run in selected),
            "second_block": {"tasks": sum(row["position"] >= 8 for row in rows),
                "solved": sum(row["position"] >= 8 and row["mse"] <= 1e-6 for row in complete),
                "costs": {key: sum(row["costs"][key] for row in rows if row["position"] >= 8)
                    for key in ("observation_cost", "measurement_cost", "compute_cost")}},
            "rejected_transfer_trials": sum(row["rejected_transfer_trials"] for row in rows)}
    common, expanded = {}, {}
    extra = output / "additional_controls.json"
    arms = [*protocol["arms"], *(json.loads(extra.read_text())["arms"] if extra.exists() else [])]
    for arm in arms:
        selected = [run for run in runs if run["arm"] == arm]
        if selected:
            common[arm] = metrics([run for run in selected if run["seed"] == protocol["code_seed"]])
            expanded[arm] = metrics(selected)
    code_runs = [run for run in runs if run["arm"] == "ordinary_code"]
    code_costs = {key: (None if any(run["costs"].get(key) is None for run in code_runs)
                       else sum(run["costs"][key] for run in code_runs))
                  for key in ("model_calls", "reported_tokens", "python_execution_seconds")}
    matched = {}
    secondary_summary = None
    if code_runs:
        primary_scores = {row["position"]: row["mse"] for row in code_runs[0]["tasks"]
                          if row["mse"] is not None}
        score_sets = {"primary_delivered_models": primary_scores}
        secondary_path = output / "secondary_authored_models.json"
        if secondary_path.exists():
            secondary = json.loads(secondary_path.read_text())
            combined = dict(primary_scores)
            combined.update({row["task_position"]: row["mse"] for row in secondary["tasks"]
                             if row["mse"] is not None})
            score_sets["including_posthoc_authored_models"] = combined
            secondary_summary = {key: secondary[key] for key in (
                "analysis", "extra_python_seconds", "extra_training_scored_rows",
                "extra_evaluation_observations", "extra_evaluation_measurement_cost")}
            secondary_summary.update(evaluated_models=len(combined),
                solved=sum(value <= 1e-6 for value in combined.values()),
                mean_mse_evaluated=mean(combined.values()) if combined else None)
        for label, code_scores in score_sets.items():
            selected = [run for run in runs if run["seed"] == protocol["code_seed"]
                        and run["arm"] != "ordinary_code"]
            positions = set(code_scores)
            for run in selected:
                positions &= {row["position"] for row in run["tasks"] if row["mse"] is not None}
            errors = {"ordinary_code": [code_scores[position] for position in sorted(positions)]}
            errors.update({run["arm"]: [row["mse"] for row in run["tasks"]
                                        if row["position"] in positions] for run in selected})
            matched[label] = {"positions": sorted(positions), "tasks": len(positions),
                "interpretation": "Prediction only on common executable models; coverage and all-task learning costs remain in common_first_seed.",
                "arms": {arm: {"solved": sum(value <= 1e-6 for value in values),
                    "mean_mse": mean(values) if values else None} for arm, values in errors.items()}}
    write(output / "summary.json", {"runs": runs, "common_first_seed": common,
        "ordinary_code_additional_costs": code_costs,
        "ordinary_code_secondary_authored_models": secondary_summary,
        "matched_prediction_comparison": matched,
        "all_available_seeds": expanded,
        "actual_learning_observations": sum(run["costs"]["observation_cost"] for run in runs),
        "actual_evaluation_observations": sum(run["evaluation_rows"] for run in runs),
        "new_learning_observations": sum(run["costs"]["observation_cost"] for run in runs if not run["reused_from"]),
        "new_evaluation_observations": sum(run["evaluation_rows"] for run in runs if not run["reused_from"]),
        "completed_sequences": sum(len(run["tasks"]) == 16 and run["evaluation_rows"] > 0 for run in runs)})
    print(json.dumps({"common": common, "all_seeds": expanded}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "research", "code", "passive", "summary"), default="prepare")
    args = parser.parse_args()
    protocol = prepare(args.output)
    if args.phase == "research":
        research(args.output, protocol)
    elif args.phase == "code":
        code(args.output, protocol)
    elif args.phase == "passive":
        passive(args.output, protocol)
    elif args.phase == "summary":
        summarize(args.output, protocol)
    else:
        print("Prepared three seeds / four research arms and one seed / ordinary Code Agent, 16 tasks each.")
