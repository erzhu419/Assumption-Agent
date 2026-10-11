#!/usr/bin/env python3
"""Ordinary LLM Python versus complete Hegel research on dynamic instruments."""

import argparse
import importlib
import json
from pathlib import Path
import sys
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/dynamic_code_comparison_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
ARMS = ("code", "hegel")
COSTS = ("observation_cost", "measurement_cost", "compute_cost")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_runtime(source):
    sys.path.insert(0, str(source.resolve()))
    for module_name, names in {
        "models": ("ResearchConfig",),
        "unified_session": ("UnifiedResearchSession", "laboratory_for"),
        "unified_code_control": ("UnifiedCodeSession",),
        "code_agent": ("CodeAgentConfig", "CodexCodeProvider"),
    }.items():
        module = importlib.import_module(f"hegel_machine.cognition.{module_name}")
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise RuntimeError(f"loaded {module.__file__}, expected {source}")
        globals().update({name: getattr(module, name) for name in names})


def tasks_for(seed, protocol):
    return [{"environment": "dynamic_instruments", "task_index": stage,
             "seed": seed + stage * protocol["stage_seed_offset"], "apparatus_seed": seed,
             "profile": "development"} for stage in range(len(protocol["stages"]))]


def new_session(arm, directory, protocol, seed, *, state=None, lab_factory=None):
    if arm == "hegel":
        return UnifiedResearchSession(ResearchConfig(**protocol["hegel_config"], seed=seed),
            tasks=tasks_for(seed, protocol), measurement_budget=protocol["code_config"]["measurement_budget"],
            state=state, lab_factory=lab_factory)
    return UnifiedCodeSession(directory / "workspace", tasks=tasks_for(seed, protocol),
        config=CodeAgentConfig(**protocol["code_config"]), state=state, lab_factory=lab_factory,
        provider_factory=lambda path: CodexCodeProvider(path, **protocol["provider"]))


def learn(arm, directory, protocol, seed, source, max_tasks=None):
    path = directory / f"{arm}_checkpoint.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    state = saved.get("session")
    if state and len(state["results"]) == len(state["tasks"]):
        print(f"{arm}: learning already complete; retained existing checkpoint", flush=True)
        return saved
    started, previous_seconds = monotonic(), saved.get("learning_wall_seconds", 0.)
    laboratories = []

    def laboratory(item):
        result = laboratory_for(item)
        laboratories.append(result)
        return result

    def checkpoint(current):
        payload = {"arm": arm, "seed": seed, "runtime_source": str(source.resolve()),
            "learning_wall_seconds": previous_seconds + monotonic() - started, "session": current}
        save(path, payload)
        active = current.get("active")
        events = active["agent_state"]["events"] if active else []
        print(json.dumps({"arm": arm, "seed": seed, "tasks_completed": len(current["results"]),
            "status": current["status"], "costs": current["costs"],
            "action": events[-1].get("action") if events else None}), flush=True)
        return payload

    session = new_session(arm, directory, protocol, seed, state=state, lab_factory=laboratory)
    visited = 0
    while len(session.state["results"]) < len(session.state["tasks"]) and (max_tasks is None or visited < max_tasks):
        task_started = monotonic()
        state = session.run(max_tasks=1, on_checkpoint=checkpoint)
        state["results"][-1]["actual_costs"] = laboratories[-1].costs
        state["results"][-1]["learning_wall_seconds"] = monotonic() - task_started
        saved = checkpoint(state)
        visited += 1
        # Each arm restores all its own knowledge after each completed task.
        session = new_session(arm, directory, protocol, seed,
            state=json.loads(json.dumps(state)), lab_factory=laboratory)
        if state["status"] in {"provider_failed", "token_usage_unavailable"}:
            break
    return saved


def evaluate(directory, protocol, seed):
    destination = directory / "comparison.json"
    if destination.exists():
        return json.loads(destination.read_text())
    reports = {arm: json.loads((directory / f"{arm}_checkpoint.json").read_text()) for arm in ARMS}
    if any(len(report["session"]["results"]) != len(report["session"]["tasks"]) for report in reports.values()):
        raise ValueError("both complete learning sequences must stop before independent evaluation")
    terminal = {key: protocol["terminal"][key] for key in ("length", "trajectories", "seed_offset")}
    for arm, report in reports.items():
        session = new_session(arm, directory, protocol, seed, state=report["session"])
        session.evaluate(**terminal)
        report["session"] = session.to_dict()
    result = {"seed": seed, "status": "evaluated", "arms": reports}
    save(destination, result)
    return result


def summarize(comparison, protocol):
    arms = {}
    for arm, report in comparison["arms"].items():
        state = report["session"]
        evaluations = {row["task_position"]: row for row in state.get("evaluation") or []}
        rows = []
        for position, entry in enumerate(state["results"]):
            research = entry["research"] if arm == "hegel" else entry
            score = evaluations.get(position, {})
            loss = score.get("loss")
            rows.append({"task_position": position, "stage": protocol["stages"][position],
                "status": research["status"], "model_origin": research.get("model_origin"),
                "usable_model": bool(research.get("program") if arm == "hegel" else research.get("model")),
                "selected_channels": research["selected_channels"], "costs": research["costs"],
                "actual_costs": entry["actual_costs"], "learning_wall_seconds": entry["learning_wall_seconds"],
                "loss": loss, "mse": score.get("mse"), "solved": loss is not None and loss <= protocol["terminal"]["solved_threshold"],
                "terminal_status": score.get("status", "completed" if score else "not_evaluated"),
                "terminal_costs": {key: score.get(key) for key in COSTS},
                "reused_files": research.get("reused_files"),
                "actions": [event["action"] for event in research.get("actions", research.get("events", []))]})
        losses = [row["loss"] for row in rows if row["loss"] is not None]
        arm_summary = {"status": state["status"], "visited_tasks": len(rows),
            "pending_tasks": len(state["tasks"]) - len(rows), "evaluated_models": len(losses),
            "solved": sum(row["solved"] for row in rows), "usable_models": sum(row["usable_model"] for row in rows),
            "mean_loss": sum(losses) / len(losses) if len(losses) == len(state["tasks"]) else None,
            "max_loss": max(losses) if losses else None,
            "measurement_free_solved": sum(row["solved"] and not row["selected_channels"] for row in rows),
            "learning_costs": state["costs"], "actual_queries": sum(row["actual_costs"]["queries"] for row in rows),
            "learning_wall_seconds": report["learning_wall_seconds"],
            "terminal_costs": {key: sum(row["terminal_costs"][key] for row in rows)
                if rows and all(row["terminal_costs"][key] is not None for row in rows) else None for key in COSTS},
            "tasks": rows}
        if arm == "code":
            usage = [event["usage"] for entry in state["results"] for event in entry["events"] if event.get("usage")]
            arm_summary["provider_usage"] = {key: sum(row[key] for row in usage)
                if usage and all(row.get(key) is not None for row in usage) else None
                for key in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens", "wall_time_seconds")}
            arm_summary["models_reported"] = sorted({row["model"] for row in usage if row.get("model")})
            arm_summary["fallback_evaluated_models"] = sum(row["model_origin"] == "last_successful_evaluation" for row in rows)
        arms[arm] = arm_summary
    return {"seed": comparison["seed"], "status": comparison["status"], "arms": arms,
            "compute_note": protocol["compute_note"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(*ARMS, "evaluate", "summary"))
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if args.seed not in protocol["paired_seeds"]:
        parser.error("seed is not declared in the protocol")
    directory = args.output / str(args.seed)
    source = ROOT / protocol["runtime_source"]
    if args.action in ARMS:
        load_runtime(source)
        learn(args.action, directory, protocol, args.seed, source, args.max_tasks)
        return
    if args.action == "evaluate":
        load_runtime(source)
        comparison = evaluate(directory, protocol, args.seed)
    elif (directory / "comparison.json").exists():
        comparison = json.loads((directory / "comparison.json").read_text())
    else:
        comparison = {"seed": args.seed, "status": "learning_incomplete", "arms": {
            arm: json.loads(path.read_text()) for arm in ARMS
            if (path := directory / f"{arm}_checkpoint.json").exists()}}
    result = summarize(comparison, protocol)
    save(directory / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
