#!/usr/bin/env python3
"""Run the frozen eight-task CodeAgent comparison with independent checkpoints."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.code_agent import CodeAgentConfig
from hegel_machine.cognition.code_session import CodeComparisonSession
from hegel_machine.cognition.models import ResearchConfig


def save(path, value):
    write_json_checkpoint(path, value, indent=2, ensure_ascii=False, allow_nan=True)


def prepare(directory):
    path = directory / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    old = json.loads((ROOT / "artifacts/code_session_v026_completion_extension_20260909.json").read_text())
    config_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    configured = {}
    for line in (config_home / "config.toml").read_text().splitlines():
        if line.startswith("["):
            break
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() in {"model", "model_reasoning_effort"}:
                configured[key.strip()] = json.loads(value.strip())
    executable = shutil.which("codex")
    protocol = {
        "experiment": "fresh_ordinary_code_agent_and_current_hegel_v043",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": [{"task_index": index % 4, "seed": 1377001 + 10007 * index} for index in range(8)],
        "config": asdict(CodeAgentConfig(**old["code"]["config"])),
        "hegel_config": asdict(ResearchConfig(**old["hegel_config"])),
        "totals": old["code"]["totals"],
        "backend": "process",
        "runtime": {"executable": executable,
            "version": subprocess.check_output([executable, "--version"], text=True).strip(),
            "config_home": str(config_home), "configured_model": configured.get("model"),
            "configured_reasoning_effort": configured.get("model_reasoning_effort"),
            "provider_model_override": None, "provider_timeout_seconds": 120,
            "python": sys.executable, "python_version": sys.version},
        "initial_knowledge": "Both arms start empty and independently retain their own subsequent knowledge.",
        "evaluation": {"after_both_learning_sequences_stop": True,
            "length": 64, "trajectories": 4, "seed_offset": 1000003,
            "metric": "prediction_mse", "missing_model": "reported_missing_not_filled"},
        "resource_contract": "No budget extensions. Report tokens between calls; a final call may cross the remaining token cap. Python time and Hegel search operations are separate resources, not equivalent compute.",
        "provider_access": "Existing CodexCodeProvider public capabilities, acquired observations, agent-written files and own lessons only; no apparatus implementation or final evaluation labels.",
        "seed_check": "No proposed seed occurred in an existing artifact protocol before this file was created.",
    }
    save(path, protocol)
    return protocol


def new_session(directory, protocol, state=None):
    return CodeComparisonSession(directory / "workspace", tasks=protocol["tasks"],
        config=protocol["config"], hegel_config=protocol["hegel_config"],
        totals=protocol["totals"], backend=protocol["backend"], state=state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("prepare", "code", "hegel", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/code_agent_fresh_v043_20260910")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol = prepare(directory)
    if args.arm == "prepare":
        print(json.dumps(protocol, ensure_ascii=False), flush=True)
        return
    checkpoint = directory / (args.arm + "_checkpoint.json")
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    session = new_session(directory, protocol, saved)

    def checkpoint_arm(_):
        state = session.to_dict()
        save(checkpoint, state)
        if args.arm == "code":
            active = state["code"]["active"]
            events = active["agent_state"]["events"] if active else []
            event = events[-1] if events else {}
            print(json.dumps({"arm": "code", "tasks_completed": len(state["code"]["results"]),
                "action": event.get("action"), "costs": state["code"]["costs"],
                "latest_usage": event.get("usage")}), flush=True)
        else:
            print(json.dumps({"arm": "hegel", "tasks_completed": len(state["hegel"]["results"]),
                "costs": state["hegel"]["costs"]}), flush=True)

    if args.arm == "code":
        session.code.run(on_checkpoint=checkpoint_arm)
        checkpoint_arm(None)
    elif args.arm == "hegel":
        session.run_hegel(on_checkpoint=checkpoint_arm)
        checkpoint_arm(None)
    else:
        code = json.loads((directory / "code_checkpoint.json").read_text())
        hegel = json.loads((directory / "hegel_checkpoint.json").read_text())
        code["hegel"] = hegel["hegel"]
        session = new_session(directory, protocol, code)
        session.evaluate()
        state = session.to_dict()
        save(directory / "comparison.json", state)
        usage = [event.get("usage", {}) for result in state["code"]["results"] for event in result["events"]]
        summary = {**state["summary"], "status": state["status"], "runtime": protocol["runtime"],
            "code_usable_models": sum(bool(result["model"]) for result in state["code"]["results"]),
            "code_task_statuses": [result["status"] for result in state["code"]["results"]],
            "model_usage": {key: sum(item.get(key) or 0 for item in usage) for key in
                ("input_tokens", "output_tokens", "cached_input_tokens", "wall_time_seconds")},
            "mean_mse": {arm: (sum(row[arm]["mse"] for row in state["evaluation"]) / len(state["evaluation"])
                if all(row[arm].get("mse") is not None for row in state["evaluation"]) else None)
                for arm in ("code", "hegel")}}
        summary["task_results"] = []
        for row in state["evaluation"]:
            index = row["task_position"]
            code_result = state["code"]["results"][index] if index < len(state["code"]["results"]) else None
            hegel_result = state["hegel"]["results"][index]
            summary["task_results"].append({**row,
                "code_status": code_result["status"] if code_result else "not_visited",
                "code_model_origin": code_result.get("model_origin") if code_result else None,
                "code_costs": code_result["costs"] if code_result else None,
                "code_model_seconds": sum(event.get("usage", {}).get("wall_time_seconds", 0.)
                    for event in code_result["events"]) if code_result else 0.,
                "code_reused_files": code_result["reused_files"] if code_result else [],
                "hegel_costs": hegel_result["costs"],
                "hegel_python_seconds": hegel_result["python_execution_seconds"]})
        save(directory / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
