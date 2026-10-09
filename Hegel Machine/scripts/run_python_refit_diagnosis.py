#!/usr/bin/env python3
"""Compare frozen Python hypotheses with their own fitting procedure on new tasks."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from random import Random
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
    execute_episode, prepare_episode, protocol as cycle_protocol, score_episode, write_json,
)
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription

SOURCE = ROOT / "artifacts/candidate_retention_cycle_luna_20261006"


def protocol(mode):
    plan = cycle_protocol()
    plan.update(study="python_same_source_refit_diagnosis", mode=mode, conditions=[],
        cases=[{"apparatus_seed": 791006001 + 1009 * i,
                "changed_response_seed": 931106001 + 1009 * i,
                "seed": (941106001 if mode == "offline" else 951106001) + 1009 * i}
               for i in range(2)],
        target_kinds=["response_parameter_change", "recurrent_instrument_change"],
        source="Reuse the two actual confirmed Luna source archives from " + str(SOURCE),
        task="Each source has two targets: finite2 with new response parameters, or recurrent "
             "with the original response parameters. Both targets share paid prefix controls. "
             "Private family labels and seeds appear only in the experiment protocol/results.",
        controller=("Fixed development controller: evaluate original; refit identical source on "
                    "paid training trajectories; confirm if separate validation supports it. "
                    "Otherwise buy two metered trajectories and use existing instrument/response "
                    "operators before fresh confirmation." if mode == "offline" else
                    "After the shared three-trajectory prefix, every action including fitting, "
                    "measurement and finish is chosen by the live Luna agent."),
        intervention="No code edits, automatic coefficient replacement or supplied true model. "
                     "The refit tool preserves the actual source and dependencies.",
        accounting="Existing source-learning costs reported once separately; no new source "
                   "learning. Targets include every paid observation and executed action. "
                   "No treatment comparison or advantage claim.",
        failure_policy="Keep every registered target and service failure; no replacement or retry.",
        scope="Two previously learned sources and four new target realizations; a bounded "
              "diagnostic pilot. Failed refitting does not prove model-family impossibility.")
    if mode == "offline":
        plan["limits"].update(max_model_calls=0, token_budget=0)
        plan["model"] = None
    return plan


def offline_research(payload):
    agent, lab = payload["agent"], payload["laboratory"]
    started = monotonic()

    def execute(action, arguments):
        feedback = agent._execute(action, arguments, lab)
        agent.events.append({"step": len(agent.events), "action": action,
                             "arguments": arguments, "feedback": feedback})
        return feedback

    try:
        original = payload["inherited_candidates"][0]["model"]
        execute("evaluate", {"model": original})
        refit = execute("run", {"operator": "refit", "model": original,
                                "path": "shared/refitted", "validation_records": [1, 2]})
        if refit["status"] == "refit_supported_on_validation":
            execute("finish", {"model": refit["path"]})
        else:
            rng = Random(task_for(payload["case"], 1)["seed"] + 701)
            execute("query", {"actions": [{"kind": "trajectory", "channels": ["meter"],
                "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]} for _ in range(2)]})
            execute("run", {"operator": "instrument", "path": "shared/instrument.json", "records": [3]})
            execute("run", {"operator": "assess_instrument", "instrument": "shared/instrument.json",
                            "records": [4]})
            response = execute("run", {"operator": "response", "instrument": "shared/instrument.json",
                                       "output_dir": "shared/revised", "validation_records": [1, 2]})
            candidates = [item for item in response["candidates"] if item["validation_mse"] is not None]
            if candidates:
                chosen = min(candidates, key=lambda item: item["validation_mse"])
                execute("finish", {"model": chosen["path"]})
        result = agent.result("completed" if agent.model else "research_unresolved")
    except Exception:
        result = agent.result("execution_failed")
        result["execution_error"] = traceback.format_exc()
    payload.update(training=result, research_wall_seconds=monotonic() - started)
    write_json(payload["path"] / "training.json", result)
    return payload


def run_study(output, *, mode="offline", provider_factory=None, auth_mode=None, study_protocol=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = deepcopy(study_protocol) if study_protocol is not None else protocol(mode)
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_candidate_retention_cycle_llm.py",
                 "run_candidate_retention_cycle.py", "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    sources, jobs = [], []
    for index, case in enumerate(plan["cases"]):
        path = SOURCE / f"case_{case['apparatus_seed']}" / "source/episode_000"
        candidate = json.loads((path / "published_candidate.json").read_text())["candidate"]
        previous = json.loads((path / "evaluation.json").read_text())
        sources.append({"path": str(path), "candidate": candidate, "costs": previous["costs"]})
        for kind in plan["target_kinds"]:
            jobs.append((case, kind, candidate, output / f"trial_{len(jobs):03d}"))

    def train(job):
        case, kind, candidate, path = job
        settings = task_for(case, 1)
        if kind == "recurrent_instrument_change":
            settings.update(instrument_mode="recurrent", response_seed=case["apparatus_seed"])
        lab = TemporalInquiryLab(**settings)
        provider = provider_factory(path) if mode == "live" else None
        payload = prepare_episode(case, 1, path, [deepcopy(candidate)], plan,
                                  provider=provider, laboratory=lab)
        return (kind, execute_episode(payload) if mode == "live" else offline_research(payload))

    with ThreadPoolExecutor(max_workers=2) as pool:
        episodes = list(pool.map(train, jobs))
    # No target's terminal labels exist until every target has stopped learning.
    results = []
    for kind, payload in episodes:
        result = score_episode(payload, plan)
        result.update(kind=kind, workspace=str(payload["path"]),
                      operator_calls=dict(Counter(event["arguments"]["operator"]
                        for event in payload["training"]["events"]
                        if event.get("arguments", {}).get("operator"))),
                      refit_actions=[event for event in payload["training"]["events"]
                        if event.get("arguments", {}).get("operator") == "refit"])
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
    result = run_study(args.output, mode=args.mode, provider_factory=factory, auth_mode=auth)
    print(json.dumps({key: value for key, value in result.items() if key not in {"sources", "results"}}, indent=2))


if __name__ == "__main__":
    main()
