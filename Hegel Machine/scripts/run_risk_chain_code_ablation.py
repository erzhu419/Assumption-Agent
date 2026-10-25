#!/usr/bin/env python3
"""Run a small offline Hegel/ordinary-Code risk-chain comparison.

The Hegel arms are offline Python. Code arms are opt-in because they call the
configured model provider. No network request is made by ``prepare`` or
``summary``.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_text_incrementally
CONFIG = ROOT / "config/risk_chain_code_ablation_20260920.json"
ARMS = ("hegel_full", "hegel_no_memory", "code_full", "code_no_memory")
sys.path.insert(0, str(ROOT / "src"))


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol(config):
    from hegel_machine.cognition.unified_session import task_sequence
    tasks = task_sequence(config["seed"], order=tuple(config["task_order"]))
    return {**config, "tasks": tasks}


def prepare(output, config=CONFIG):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    value = protocol(load_config(config))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("frozen protocol differs; use a new output directory")
    if not path.exists():
        path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def _run_hegel(output, p, arm, *, evaluate=False):
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    settings = {**p["research_config"], "seed": p["seed"],
                "use_memory": arm == "hegel_full"}
    path = output / f"{arm}.json"
    state = json.loads(path.read_text()) if path.exists() else None
    session = UnifiedResearchSession(ResearchConfig(**settings), tasks=p["tasks"],
        measurement_budget=p["measurement_budget"], state=state)
    session.run()
    if evaluate:
        session.evaluate(**p["evaluation"])
    write_text_incrementally(path, json.dumps(session.to_dict(), indent=2) + "\n")


def _run_code(output, p, arm, *, evaluate=False):
    from hegel_machine.cognition.unified_code_control import UnifiedCodeSession
    from hegel_machine.cognition.code_agent import CodeAgentConfig
    from hegel_machine.cognition.responses_code_provider import ResponsesCodeProvider
    workspace = output / arm
    config = CodeAgentConfig(**p["code_config"])
    if not os.environ.get("RUOLI_API_KEY"):
        raise RuntimeError("--run-code requires RUOLI_API_KEY in the environment")
    state_path = output / f"{arm}.json"
    saved = json.loads(state_path.read_text()) if state_path.exists() else None

    def provider(path):
        return ResponsesCodeProvider(path, base_url=os.environ.get("RUOLI_BASE_URL", "https://ruoli.dev/v1"),
            model=p["model"], timeout_seconds=p["provider_timeout_seconds"],
            reasoning_effort=p["reasoning_effort"], max_output_tokens=p["max_output_tokens"])

    session = UnifiedCodeSession(workspace, tasks=p["tasks"], config=config,
        state=saved, provider_factory=provider, use_memory=arm == "code_full")

    def checkpoint(state):
        write_text_incrementally(state_path, json.dumps(state, indent=2) + "\n")

    try:
        while len(session.state["results"]) < len(session.state["tasks"]):
            session.run(max_tasks=1, on_checkpoint=checkpoint)
            checkpoint(session.to_dict())
            if session.state.get("status") in {"provider_failed", "token_usage_unavailable"}:
                return
    except Exception as error:
        # The active state and actual accumulated costs are already in the
        # checkpoint. Keep the failure visible and let the other arms finish.
        session.state["status"] = "runner_failed"
        session.state["runner_error"] = f"{type(error).__name__}: {error}"
        checkpoint(session.to_dict())
        return
    if len(session.state["results"]) != len(session.state["tasks"]):
        return
    if evaluate:
        session.evaluate(**p["evaluation"])
    checkpoint(session.to_dict())


def run(output, *, run_code=False):
    output = Path(output)
    p = json.loads((output / "protocol.json").read_text())
    _run_hegel(output, p, "hegel_full")
    _run_hegel(output, p, "hegel_no_memory")
    if run_code:
        with ThreadPoolExecutor(max_workers=p.get("concurrent_code_arms", 2)) as pool:
            jobs = [pool.submit(_run_code, output, p, arm) for arm in ("code_full", "code_no_memory")]
            for job in jobs:
                job.result()
    # Finish learning for every available arm before independent evaluation.
    for arm in ARMS:
        path = output / f"{arm}.json"
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if arm.startswith("hegel"):
            from hegel_machine.cognition.unified_session import UnifiedResearchSession
            session = UnifiedResearchSession(state=state)
        else:
            from hegel_machine.cognition.unified_code_control import UnifiedCodeSession
            session = UnifiedCodeSession(output / arm, state=state)
        if len(session.state["results"]) == len(session.state["tasks"]):
            session.evaluate(**p["evaluation"])
            write_text_incrementally(path, json.dumps(session.to_dict(), indent=2) + "\n")


def summary(output):
    output = Path(output)
    result = {"experiment": "risk_chain_code_ablation", "arms": {}}
    for arm in ARMS:
        path = output / f"{arm}.json"
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        rows = state.get("results", [])
        scores = state.get("evaluation") or []
        result_statuses = [item.get("status", item.get("research", {}).get("status", "unknown"))
                           for item in rows]
        score_statuses = [item.get("status", "evaluated" if item.get("loss") is not None else "missing_loss")
                          for item in scores]
        valid = [item["loss"] for item in scores if item.get("loss") is not None]
        planned = len(state.get("tasks", rows))
        result["arms"][arm] = {
            "tasks": len(rows), "planned_tasks": planned, "evaluated": len(scores),
            "costs": state.get("costs"),
            "loss_kind": sorted({item["loss_kind"] for item in scores}),
            "mean_loss": sum(valid) / planned if planned > 0 and len(valid) == planned else None,
            "observed_mean_loss": sum(valid) / len(valid) if valid else None,
            "valid_loss_scores": len(valid),
            "evaluation_coverage": len(valid) / planned if planned else 0.,
            "result_status_counts": {status: result_statuses.count(status)
                                     for status in sorted(set(result_statuses))},
            "evaluation_status_counts": {status: score_statuses.count(status)
                                         for status in sorted(set(score_statuses))},
        }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "summary"), required=True)
    parser.add_argument("--run-code", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "run":
        run(args.output, run_code=args.run_code)
    else:
        print(json.dumps(summary(args.output), indent=2))
