#!/usr/bin/env python3
"""Replay the four Python diagnosis tasks with research_choice-driven routing."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts.run_candidate_retention_cycle import task_for
from scripts.run_candidate_retention_cycle_llm import (
    execute_episode, prepare_episode, score_episode, write_json,
)
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription
from scripts.run_python_refit_diagnosis import SOURCE, protocol as diagnosis_protocol


def protocol(mode):
    # Keep the live pilot's actual four tasks and paid-prefix seeds in both modes.
    plan = diagnosis_protocol("live")
    plan.update(study="python_research_choice_routing_development_replay", mode=mode,
        controller=("At most 12 research_choice invocations, each followed only by its returned "
                    "next_action. The first choice receives the actual archived source path; "
                    "subsequent choices use their default candidate handling. The controller "
                    "does not inspect MSE or private task kind to select research actions. "
                    "Stop after confirmed submission, no next action, or 12 choice rounds. "
                    "Twelve offline rounds allow 12 choices plus at most 12 returned actions; "
                    "they are not the live arm's 12 model calls." if mode == "offline" else
                    "After the common three-trajectory prefix, Luna selects all actions through "
                    "the unchanged agent loop, including whether to use research_choice and "
                    "follow its recommendation. At most 12 model calls; no host research actions."),
        scope="Development replay of the same two source apparatuses and four target tasks "
              "previously used by the live Python-refit pilot and threshold-disclosure replay. "
              "These are not new independent samples. Offline routing is a deterministic "
              "recommendation consumer, not evidence of autonomous LLM adoption.",
        offline_choice_round_limit=12,
        previous_experiment="artifacts/python_refit_diagnosis_luna_20261006",
        intervention="Expose existing refitting and measured instrument/response research "
                     "directions through Python research_choice. The host does not select a "
                     "repair using hidden family labels, numerical error, or terminal outcomes.")
    if mode == "offline":
        plan["limits"].update(max_model_calls=0, token_budget=0)
        plan["model"] = None
    return plan


def routing_research(payload, *, max_rounds=12):
    agent, lab = payload["agent"], payload["laboratory"]
    started = monotonic()

    def execute(action, arguments):
        feedback = agent._execute(action, arguments, lab)
        agent.events.append({"step": len(agent.events), "action": action,
                             "arguments": arguments, "feedback": feedback})
        return feedback

    status = "routing_round_limit"
    try:
        for index in range(max_rounds):
            arguments = {"operator": "research_choice"}
            if index == 0:
                arguments["models"] = [payload["inherited_candidates"][0]["model"]]
            decision = execute("run", arguments)
            action = decision.get("next_action")
            if action is None:
                status = "research_unresolved"
                break
            execute(action["action"], action.get("arguments", {}))
            if agent.model:
                status = "completed"
                break
        result = agent.result(status)
    except Exception:
        result = agent.result("execution_failed")
        result["execution_error"] = traceback.format_exc()
    payload.update(training=result, research_wall_seconds=monotonic() - started)
    write_json(payload["path"] / "training.json", result)
    return payload


def operator_calls(events):
    """Count executed operators, including continuations after paid queries."""
    counts = Counter()

    def continuations(feedback):
        for result in feedback.get("after_observation_results", []):
            if result.get("operator"):
                counts[result["operator"]] += 1
            continuations(result)

    for event in events:
        operator = event.get("arguments", {}).get("operator")
        if event.get("action") == "run" and operator:
            counts[operator] += 1
        continuations(event.get("feedback", {}))
    return dict(counts)


def run_study(output, *, mode="offline", provider_factory=None, auth_mode=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = protocol(mode)
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_python_refit_diagnosis.py", "run_candidate_retention_cycle_llm.py",
                 "run_candidate_retention_cycle.py", "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    sources, jobs = [], []
    for case in plan["cases"]:
        path = SOURCE / f"case_{case['apparatus_seed']}" / "source/episode_000"
        candidate = json.loads((path / "published_candidate.json").read_text())["candidate"]
        previous = json.loads((path / "evaluation.json").read_text())
        sources.append({"path": str(path), "candidate": candidate, "costs": previous["costs"]})
        offset = len(jobs)
        jobs.extend((case, kind, candidate, output / f"trial_{offset + index:03d}")
                    for index, kind in enumerate(plan["target_kinds"]))

    def train(job):
        case, kind, candidate, path = job
        settings = task_for(case, 1)
        if kind == "recurrent_instrument_change":
            settings.update(instrument_mode="recurrent", response_seed=case["apparatus_seed"])
        payload = prepare_episode(case, 1, path, [deepcopy(candidate)], plan,
            provider=provider_factory(path) if mode == "live" else None,
            laboratory=TemporalInquiryLab(**settings))
        return kind, (execute_episode(payload) if mode == "live" else
                      routing_research(payload, max_rounds=plan["offline_choice_round_limit"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        episodes = list(pool.map(train, jobs))
    # Finish every learning episode before producing any target terminal labels.
    results = []
    for kind, payload in episodes:
        result = score_episode(payload, plan)
        events = payload["training"]["events"]
        result.update(kind=kind, workspace=str(payload["path"]),
            operator_calls=operator_calls(events),
            research_choices=[event for event in events
                              if event.get("arguments", {}).get("operator") == "research_choice"],
            refit_actions=[event for event in events if event.get("arguments", {}).get("operator") == "refit"])
        if mode == "offline":
            result["missing_usage_calls"] = 0
        write_json(payload["path"] / "evaluation.json", result)
        results.append(result)
    summary = {"study": plan["study"], "mode": mode, "sources": sources, "results": results,
        "registered_targets": len(jobs), "confirmed_and_correct": sum(r["confirmed_and_correct"] for r in results),
        "false_confirmations": sum(r["false_confirmation"] for r in results),
        "existing_source_costs": total_costs([source["costs"] for source in sources]),
        "new_target_costs": total_costs([r["costs"] for r in results]),
        **{key: sum(r[key] for r in results) for key in
           ("known_tokens", "input_tokens", "cached_input_tokens", "missing_usage_calls")},
        "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = verify_codex_subscription() if args.mode == "live" else None
    factory = lambda path: CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium",
                                             timeout_seconds=180)
    summary = run_study(args.output, mode=args.mode, provider_factory=factory, auth_mode=auth)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"sources", "results"}}, indent=2))


if __name__ == "__main__":
    main()
