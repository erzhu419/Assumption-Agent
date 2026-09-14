#!/usr/bin/env python3
"""Run the frozen eight-task ordinary-code study in separate research phases."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.code_agent import CodeAgentConfig, CodexCodeProvider
from hegel_machine.cognition.code_session import CodeComparisonSession
from hegel_machine.cognition.models import ResearchConfig

TASKS = [{"task_index": index % 4, "seed": 18313 + 10007 * index} for index in range(8)]
CHECKPOINT = ROOT / "artifacts/code_session_v026_20260909.json"
WORKSPACE = ROOT / "artifacts/code_session_v026_20260909_workspace"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("first", "second", "hegel"))
    args = parser.parse_args()
    if args.phase == "first":
        if CHECKPOINT.exists():
            raise FileExistsError("The frozen study already has a checkpoint")
        session = CodeComparisonSession(WORKSPACE, tasks=TASKS[:4],
            config=CodeAgentConfig(observation_budget=96, measurement_budget=384, compute_budget=60000,
                max_model_calls=8, max_steps=12, token_budget=100000, execution_seconds=60),
            hegel_config=ResearchConfig(observation_budget=96, compute_budget=60000, max_rounds=3),
            totals={"observation_budget": 768, "measurement_budget": 3072, "compute_budget": 480000,
                    "max_model_calls": 64, "token_budget": 400000, "execution_seconds": 240},
            provider_factory=lambda path: CodexCodeProvider(path, timeout_seconds=180))
    else:
        state = json.loads(CHECKPOINT.read_text())
        hegel_state = ROOT / "artifacts/code_session_v026_hegel_learning_20260909.json"
        if args.phase == "hegel" and hegel_state.exists():
            state["hegel"] = json.loads(hegel_state.read_text())
        session = CodeComparisonSession(WORKSPACE, state=state,
            provider_factory=lambda path: CodexCodeProvider(path, timeout_seconds=180))
        if args.phase == "second" and len(session.code.state["tasks"]) == 4:
            with (ROOT / "artifacts/code_session_v026_first_four_20260909.json").open("x") as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2)
            for task in TASKS[4:]:
                session.add_task(**task)

    def checkpoint(state=None):
        payload = state or session.to_dict()
        CHECKPOINT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        code = payload["code"]
        active = code["active"]
        print(json.dumps({"phase": args.phase, "completed": len(code["results"]),
            "active_actions": len(active["agent_state"]["events"]) if active else None,
            "costs": code["costs"], "hegel_completed": len(payload["hegel"]["results"])}), flush=True)

    if args.phase in {"first", "second"}:
        session.code.run(on_checkpoint=lambda _: checkpoint())
    else:
        session.run(max_tasks=0, on_checkpoint=checkpoint)
        session.evaluate()
    checkpoint()


if __name__ == "__main__":
    main()
