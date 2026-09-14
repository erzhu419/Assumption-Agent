#!/usr/bin/env python3
"""Compare composed research and a real ordinary Python agent on one task stream."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession

SEEDS = (3050319137, 3050419151, 3050519167)
ARMS = {"default": {}, "no_memory": {"use_memory": False},
        "no_program_synthesis": {"max_question_depth": 0}}
TASKS = (
    {"task_index": 0, "profile": "development"},
    {"readout": "total"}, {"readout": "event"},
    {"readout": "total_event"}, {"readout": "filtered_total"},
    {"readout": "window_total_event"},
    {"readout": "total_event", "rename_prefix": "renamed_"},
    {"readout": "window_total_event", "rename_prefix": "later_"},
)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tasks_for(seed):
    apparatus_seed = dict(zip(SEEDS, (6131, 7141, 8161))).get(seed, 9013)
    return [{**({"environment": "objects", "objects": 4, "variant": "exchange",
                 "apparatus_seed": apparatus_seed} if "readout" in item else {}),
             **item, "seed": seed + 10007 * index} for index, item in enumerate(TASKS)]


def configuration():
    return ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True)


def prepare(output, *, seeds=None, arms=None):
    path = output / "protocol.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    seeds = list(seeds or (saved["seeds"] if saved else SEEDS))
    arms = list(arms or (saved["research_arms"] if saved else ARMS))
    protocol = {
        "experiment": "unified_composed_research_and_real_code_v062",
        "seeds": seeds, "research_arms": {name: ARMS[name] for name in arms},
        "tasks_by_seed": {str(seed): tasks_for(seed) for seed in seeds},
        "research_config": asdict(configuration()), "measurement_budget": 4608,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "deterministic_solved_threshold": 1e-6, "code_seed": seeds[0],
        "code_config": {"observation_budget": 384, "measurement_budget": 4608,
            "compute_budget": 360000, "execution_seconds": 60., "max_steps": 12,
            "max_model_calls": 12, "token_budget": 240000},
        "provider_timeout_seconds": 300,
        "provider_model_override": None,
        "learning_contract": "A common public scalar/object apparatus interface, bounds, purchasable "
            "measurement views and object role schema. Readout names, physical parameters and apparatus "
            "source remain outside both learner inputs. Object tasks share a physical apparatus while "
            "changing readout, history operator and measured feature names. Task observations are new.",
        "memory_contract": "Hegel shares its learned executable concepts and research programmes; "
            "ordinary Code shares its own Python files and lessons. Code receives no Hegel source, "
            "predictions, proposed expressions, or final evaluation rows.",
        "evaluation_contract": "The entire learning stream completes before independent evaluation. "
            "Selected models are evaluated without modification on the same four fresh 64-row "
            "trajectories per task. Missing and failed models stay in all-task solved coverage; "
            "common executable model errors are a separate endpoint.",
        "cost_contract": "Observation, measurement and scored-row caps are matched. Hegel search "
            "computation and Code Python execution seconds/model calls/reported tokens are reported "
            "separately. Code token stopping is between calls, so a last call can exceed its cap.",
        "ablation_contract": "no_program_synthesis sets max_question_depth=0: it removes both "
            "new research-program proposals and the existing recursive residual question graph. "
            "It is a combined recursion/program-synthesis ablation, not their separate causal attribution.",
        "scope": "Constructed compositional tasks with supplied measurable object roles. "
            "Three research seeds, one real ordinary-LLM seed; not an open-world generality claim.",
        "planned_research_tasks": len(seeds) * len(arms) * len(TASKS),
        "planned_code_tasks": len(TASKS), "maximum_code_calls": 12 * len(TASKS),
    }
    if saved is not None and saved != protocol:
        raise ValueError("protocol changed; preserve this cohort and use a new output directory")
    if saved is None:
        write(path, protocol)
    return protocol


def freeze(output, kind):
    names = ("code_agent.py", "unified_code_control.py", "unified_object_lab.py", "structure_lab.py",
             "unknown_task_lab.py", "proposers.py", "models.py", "measurement_lab.py", "evaluation.py")
    sources = ([ROOT / "src/hegel_machine/cognition" / name for name in names] if kind == "code"
               else [path for path in (ROOT / "src/hegel_machine/cognition").glob("*.py")
                     if path.name not in {"code_agent.py", "unified_code_control.py"}])
    for source in [*sources, Path(__file__).resolve()]:
        target = output / (kind + "_source") / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"{kind} method changed during cohort: {source.name}")
        else:
            shutil.copyfile(source, target)


def research(output, protocol, seed=None, arm=None, max_tasks=None):
    freeze(output, "research")
    for selected_seed in protocol["seeds"]:
        if seed is not None and seed != selected_seed:
            continue
        for name, overrides in protocol["research_arms"].items():
            if arm is not None and name != arm:
                continue
            path = output / "runs" / f"{selected_seed}_{name}.json"
            saved = json.loads(path.read_text()) if path.exists() else None
            tasks = protocol["tasks_by_seed"][str(selected_seed)]
            if saved and len(saved["evaluation"] or []) == len(tasks):
                continue
            session = UnifiedResearchSession(ResearchConfig(**{**protocol["research_config"],
                "seed": selected_seed, **overrides}), tasks=tasks,
                measurement_budget=protocol["measurement_budget"], state=saved)

            def checkpoint(report):
                write(path, report)
                print(json.dumps({"seed": selected_seed, "arm": name,
                    "completed_tasks": len(report["results"]), "costs": report["costs"]}), flush=True)

            session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
            if len(session.state["results"]) == len(tasks):
                checkpoint(session.evaluate(**protocol["evaluation"]))


def code(output, protocol, max_tasks=None):
    from hegel_machine.cognition.code_agent import CodexCodeProvider
    from hegel_machine.cognition.unified_code_control import UnifiedCodeSession
    freeze(output, "code")
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("the configured Codex executable is unavailable")
    runtime_path = output / "code_runtime.json"
    if not runtime_path.exists():
        config_path = Path.home() / ".codex/config.toml"
        contents = config_path.read_text() if config_path.exists() else ""
        configured = {}
        for key in ("model", "model_reasoning_effort"):
            found = re.search(r'^' + key + r'\s*=\s*"([^"\n]+)"', contents, re.M)
            configured[key] = found.group(1) if found else None
        write(runtime_path, {"executable": executable,
            "version": subprocess.check_output([executable, "--version"], text=True).strip(),
            "python": sys.executable, "configured": configured, "provider_model_override": None,
            "provider_timeout_seconds": protocol["provider_timeout_seconds"],
            "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    seed = protocol["code_seed"]
    path = output / "runs" / f"{seed}_ordinary_code.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved and saved["evaluation"] is not None:
        return
    session = UnifiedCodeSession(output / "code_workspace",
        tasks=protocol["tasks_by_seed"][str(seed)], config=protocol["code_config"], state=saved,
        provider_factory=lambda path: CodexCodeProvider(path, executable=executable,
            timeout_seconds=protocol["provider_timeout_seconds"]))

    def checkpoint(report):
        write(path, report)
        print(json.dumps({"seed": seed, "arm": "ordinary_code",
            "completed_tasks": len(report["results"]),
            "active_task": report["active"]["task_position"] if report["active"] else None,
            "status": report["status"], "costs": report["costs"]}), flush=True)

    session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
    if len(session.state["results"]) == len(session.state["tasks"]):
        session.evaluate(**protocol["evaluation"])
        checkpoint(session.to_dict())


def metrics(rows, threshold):
    scored = [row for row in rows if row["loss"] is not None]
    return {"tasks": len(rows), "evaluated_models": len(scored),
        "solved": sum(row["loss"] <= threshold for row in scored),
        "mean_loss_evaluated": mean(row["loss"] for row in scored) if scored else None,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "source_uses": sum(bool(row["source_uses"]) for row in rows),
        "learning_costs": {key: sum(row["costs"][key] for row in rows)
            for key in ("observation_cost", "measurement_cost", "compute_cost")}}


def summarize(output, protocol):
    runs = []
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        seed, arm = path.stem.split("_", 1)
        scores = {row["task_position"]: row for row in report["evaluation"] or []}
        rows = []
        for index, row in enumerate(report["results"]):
            result = row if arm == "ordinary_code" else row["research"]
            rows.append({"position": index, "status": result["status"],
                "readout": report["tasks"][index].get("readout", "scalar"),
                "costs": result["costs"], "loss": scores.get(index, {}).get("loss"),
                "selected_channels": result["selected_channels"],
                "selected_kind": result.get("selected_kind", result.get("model_origin")),
                "source_uses": result.get("used_sources", result.get("reused_files", [])),
                "method_source_ids": result.get("method_source_ids", []),
                "composition_actions": [event for event in result.get("actions", [])
                    if "compos" in str(event.get("action", "")) or "research_program" in str(event.get("action", ""))]})
        runs.append({"seed": int(seed), "arm": arm, "status": report["status"], "tasks": rows,
            "costs": report["costs"], "evaluated_tasks": len(scores),
            "evaluation_observations": sum(row["observation_cost"] for row in scores.values()),
            "evaluation_measurement_cost": sum(row["measurement_cost"] for row in scores.values())})
    threshold = protocol["deterministic_solved_threshold"]
    arms, common = {}, {}
    for arm in [*protocol["research_arms"], "ordinary_code"]:
        selected = [run for run in runs if run["arm"] == arm]
        rows = [row for run in selected for row in run["tasks"]]
        arms[arm] = {"all_tasks": metrics(rows, threshold),
            "composed_tasks": metrics([row for row in rows if row["position"] >= 3], threshold),
            "renamed_successors": metrics([row for row in rows if row["position"] >= 6], threshold)}
        common[arm] = metrics([row for run in selected if run["seed"] == protocol["code_seed"]
                              for row in run["tasks"]], threshold)
    code_runs = [run for run in runs if run["arm"] == "ordinary_code"]
    summary = {"experiment": protocol["experiment"], "arms": arms, "common_first_seed": common,
        "runs": runs, "planned_sequences": len(protocol["seeds"]) * len(protocol["research_arms"]) + 1,
        "completed_sequences": sum(run["evaluated_tasks"] == len(TASKS) for run in runs),
        "actual_learning_observations": sum(run["costs"]["observation_cost"] for run in runs),
        "actual_evaluation_observations": sum(run["evaluation_observations"] for run in runs),
        "ordinary_code_costs": {key: (None if any(run["costs"].get(key) is None for run in code_runs)
             else sum(run["costs"][key] for run in code_runs))
             for key in ("model_calls", "reported_tokens", "python_execution_seconds")}}
    write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", choices=tuple(ARMS), nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--max-tasks", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summary-only", action="store_true")
    mode.add_argument("--code", action="store_true")
    args = parser.parse_args()
    protocol = prepare(args.output, seeds=args.seeds, arms=args.arms)
    if args.summary_only:
        summarize(args.output, protocol)
    elif args.code:
        code(args.output, protocol, args.max_tasks)
    elif not args.prepare_only:
        research(args.output, protocol, args.seed, args.arm, args.max_tasks)


if __name__ == "__main__":
    main()
