#!/usr/bin/env python3
"""Three-task development pilot with one real default-model Codex call per task.

Run t01 first to establish transport success, then resume t02/t03. Existing
task/arm results are retained so resuming does not repeat model calls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.proposers import CodexProposer


def summarize(payload):
    rows = []
    for record in payload["runs"]:
        episode = record.get("episode", {})
        actual_calls = [action["details"]["usage"] for action in episode.get("actions", [])
                        if action["action"] == "llm_proposal"
                        and action["details"]["usage"].get("calls", 0)]
        rows.append({
            "task_id": record["task_id"], "arm": record["arm"], "status": record["status"],
            "test_loss": record.get("test_loss"), "validation_loss": episode.get("validation_loss"),
            "expression": episode.get("expression"), "selected_operation": episode.get("selected_operation"),
            "observation_cost": episode.get("observation_cost"), "compute_cost": episode.get("compute_cost"),
            "wall_time_seconds": record["wall_time_seconds"],
            "llm_calls": sum(item.get("calls", 0) for item in actual_calls),
            "llm_status": [item["status"] for item in actual_calls],
            "llm_reported_tokens": [item.get("total_tokens") for item in actual_calls],
            "llm_wall_seconds": [item.get("wall_time_seconds") for item in actual_calls],
            "models": [item.get("model") for item in actual_calls],
            "runtime_versions": [item.get("runtime_version") for item in actual_calls],
            "instruction_protocols": [item.get("instruction_protocol", "inherited_coding_instructions_v1") for item in actual_calls],
            "token_budget_exceeded": any(item.get("token_budget_exceeded", False) for item in actual_calls),
            "error": record.get("error"),
        })
    calls = [value for row in rows for value in row["llm_reported_tokens"]]
    return {
        "experiment": payload["experiment"], "partition": "development", "seed": payload["seed"],
        "runs": rows, "real_llm_calls": sum(row["llm_calls"] for row in rows),
        "reported_total_tokens": sum(value for value in calls if value is not None),
        "token_usage_complete": all(value is not None for value in calls),
        "protocol_changes": payload.get("protocol_changes", []),
        "limitations": payload["limitations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("t01", "t02", "t03"), default=["t01", "t02", "t03"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    artifact = ROOT / "artifacts" / "cognition_v2_llm_pilot.json"
    summary_path = ROOT / "artifacts" / "cognition_v2_llm_pilot_summary.json"
    base = ResearchConfig(
        seed=args.seed, partition="development", observation_budget=48, compute_budget=12000,
        max_rounds=3, max_history=4, use_memory=False, use_abstractions=False,
        llm_calls_per_task=1, llm_timeout_seconds=120, llm_token_budget=8000,
    )
    if artifact.exists():
        payload = json.loads(artifact.read_text())
        if payload["seed"] != args.seed:
            raise ValueError("existing pilot uses another seed; preserve its results before a new experiment")
    else:
        payload = {
            "experiment": "three_task_real_codex_development_pilot",
            "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "common_config": asdict(base), "runs": [],
            "limitations": [
                "Three public synthetic development tasks are an operational pilot, not independent efficacy confirmation.",
                "Both arms have the same initial observations, observation permissions, executable language and budget caps; adaptive queries may differ.",
                "One real Codex CLI call per task inherits the configured default model; no model override is supplied.",
                "8000 reported tokens is an admission budget; the CLI has no enforced per-call total-token cap and overruns are reported.",
                "Candidate-observation compute and real model tokens/wall time are separate resource measures.",
            ],
        }
    if "protocol_changes" not in payload:
        legacy = [action["details"]["usage"] for record in payload["runs"]
                  for action in record.get("episode", {}).get("actions", [])
                  if action["action"] == "llm_proposal" and action["details"]["usage"].get("calls")
                  and "instruction_protocol" not in action["details"]["usage"]]
        payload["protocol_changes"] = [{
            "after_task": "t01", "reason": "CLI inherited coding context consumed 15214 input tokens for a 5829-character task prompt",
            "change": "t02/t03 use a short model_instructions_file, zero project-doc context, empty developer_instructions and disabled legacy notify; default model is unchanged",
            "comparison_limit": "t01 and later tasks are separate operational cases with different instruction envelopes",
        }] if legacy else []
    completed = {(row["task_id"], row["arm"]) for row in payload["runs"]}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    for task_id in args.tasks:
        for arm in ("local_cognitive_search", "ordinary_codex_proposer"):
            if (task_id, arm) in completed:
                continue
            environment = next(env for env in make_suite(args.seed, partition="development")
                               if env.spec.task_id == task_id and env.evaluation_metadata["phase"] == "acquire")
            config = replace(base, proposer="codex" if arm == "ordinary_codex_proposer" else "search")
            proposer = CodexProposer(timeout_seconds=120, token_budget=8000, max_calls=1,
                                     max_candidates=4, max_probes=1, max_history=4) if arm == "ordinary_codex_proposer" else None
            print(json.dumps({"event": "start", "task_id": task_id, "arm": arm}), flush=True)
            started = perf_counter()
            record = {"task_id": task_id, "arm": arm, "config": asdict(config)}
            try:
                episode = CognitiveAgent(config, proposer=proposer).run(environment)
                final_rows = environment.evaluation_rows()
                record.update(status="completed", episode=episode.to_dict(), evaluation_rows=len(final_rows),
                              test_loss=loss(episode.program, final_rows, episode.fallback, environment.spec.risk_quantile))
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error),
                              actual_proposer_calls=proposer.calls if proposer else 0)
            record["wall_time_seconds"] = perf_counter() - started
            payload["runs"].append(record)
            write_text_incrementally(artifact, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            summary = summarize(payload)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps({"event": "complete", **summary["runs"][-1]}, ensure_ascii=False), flush=True)
    write_text_incrementally(artifact, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    summary_path.write_text(json.dumps(summarize(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"artifact": str(artifact), "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
