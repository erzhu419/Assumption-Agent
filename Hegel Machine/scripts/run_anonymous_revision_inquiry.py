#!/usr/bin/env python3
"""Development study of authored representations and prospective experiments.

All arms receive identical paid prefixes and the same optional executable archive.
The provider sees the public trajectory interface, not the simulator family or a
menu of fitted state models. Old paper cohorts and their memory arms are unchanged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.code_agent import CodexCodeProvider, evaluate_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.proposers import _events
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


def protocol(*, kinds=("memoryless", "delay", "hysteresis"), cases=1,
             arms=("code_react", "hegel"), seed=61011017, config=None,
             model="gpt-6-luna", timeout_seconds=300, archive=None):
    return {
        "study": "anonymous_authored_revision_development",
        "model": model, "arms": list(arms),
        "cases": [{"id": f"case_{index:03d}", "kind": kind, "seed": seed + index * 1009,
                   "prefix_seed": seed + index * 1009 + 100003,
                   "confirmation_seed": seed + index * 1009 + 200003,
                   "evaluation_seed": seed + index * 1009 + 300007}
                  for index, kind in enumerate(kind for _ in range(cases) for kind in kinds)],
        "limits": asdict(config or InquiryLLMConfig(observation_budget=576,
            confirmation_rows=96, max_model_calls=48, token_budget=1200000,
            execution_seconds=300., compute_budget=250000)),
        "provider": {"backend": "codex_cli", "timeout_seconds": timeout_seconds,
                     "reasoning_effort": "medium"},
        "prefix_rows": 72, "archive": str(archive) if archive else None,
        "controls": "Both arms have the same code archive, paid prefix, Python and experiment tools. "
                    "Only Hegel requires an executable hypothesis/research-question manifest.",
        "discovery": "No supplied state-model menu, diagnostic label, fitted candidate, or research procedure. "
                     "Predictors, representations, experiment programs and readouts are provider-authored.",
        "evaluation": {"rows": 512, "trajectory_length": 48, "mse_threshold": 1e-10},
        "terminal_order": "All training stops before terminal labels are generated. "
                          "Submitted predictor files and their paid training view stay fixed.",
        "failure_policy": "Keep all initially registered sources and provider, research and confirmation failures.",
        "scope": "Deterministic development controls on 21 public input levels; learning horizon 24. "
                 "Terminal horizon 48 tests new paths, not different physical objects. "
                 "This pilot alone establishes neither recursive superiority nor cross-task meta-prior learning.",
    }


def restore_pending_question_error(agent, laboratory, previous, workspace, *, model):
    """Return the saved, paid declaration error without rerunning scientific work."""
    workspace = Path(workspace)
    call = previous['costs']['model_calls'] + 1
    reply = workspace / f'code_action_{call:03d}.json'
    trace = workspace / f'code_call_{call:03d}.jsonl'
    payload = json.loads(reply.read_text())
    arguments = json.loads(payload['arguments_json'], strict=False)
    if payload.get('source_code') is not None:
        arguments['code'] = payload['source_code']
    if payload.get('research_question') is not None:
        arguments['research_question'] = payload['research_question']
    if 'research_notes' in payload:
        arguments['notes'] = payload['research_notes']
    declaration = arguments.get('research_question') or arguments.get('question')
    if (payload['action'] != 'evaluate' or arguments.get('operator') is not None
            or 'component' in arguments or arguments.get('target', 'response') != 'response'
            or not isinstance(declaration, str) or not declaration
            or previous.get('online_questions') or previous.get('model') is not None):
        raise ValueError('recover only the saved pre-execution question-declaration failure')
    events, _ = _events(trace.read_text())
    completion = next((event for event in reversed(events)
                       if event.get('type') == 'turn.completed'), None)
    if completion is None:
        raise ValueError('the pending paid reply has no completed usage record')
    usage = {'backend': 'codex_cli', 'model': model, 'calls': 1,
             **{name: completion['usage'].get(name)
                for name in ('input_tokens', 'output_tokens', 'cached_input_tokens')}}
    if not all(isinstance(usage[key], int) for key in ('input_tokens', 'output_tokens')):
        raise ValueError('the pending paid reply has no reported token usage')
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
    agent.costs['model_calls'] = call
    agent.costs['reported_tokens'] += usage['total_tokens']
    try:
        agent._execute(payload['action'], arguments, laboratory)
    except (ValueError, TypeError) as error:
        feedback = {'error': f'{type(error).__name__}: {error}'}
    else:
        raise RuntimeError('malformed question must return before scientific execution')
    agent.events.append({'step': call - 1, 'action': payload['action'],
                         'arguments': arguments, 'feedback': feedback, 'usage': usage})
    agent._after_model_turn(laboratory)
    agent.provider.calls = call
    return {'saved_reply': str(reply), 'saved_usage': str(trace), 'model_call': call,
            'reported_tokens': usage['total_tokens'],
            'scope': 'Saved paid reply charged once and returned as ordinary input feedback. '
                'No predictor write, evaluation, acquisition or extra allowance.'}


def run_study(plan, output, *, provider_factory, knowledge=None, prepare_prefix=None,
              laboratory_factory=None, workers=1, agent_factory=CommonArchiveInquiryAgent,
              previous_study=None, resume_task=''):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a fresh output directory to retain earlier developmental failures")
    previous_study = Path(previous_study).resolve() if previous_study else None
    if previous_study:
        previous_plan = json.loads((previous_study / 'protocol.json').read_text())
        for field in ('cases', 'arms', 'model', 'provider', 'limits', 'prefix_rows', 'evaluation'):
            if plan[field] != previous_plan[field]:
                raise ValueError(f'cohort recovery must preserve registered {field}')
    write_json(output / "protocol.json", plan)
    # Literal common input is saved once; no arm learns the other's live results.
    write_json(output / "initial_archive.json", knowledge or {})
    limits = InquiryLLMConfig(**plan["limits"])
    if laboratory_factory is None:
        laboratory_factory = lambda case: AnonymousHistoryLab(kind=case["kind"], seed=case["seed"])
    jobs = []
    for case_index, case in enumerate(plan["cases"]):
        arms = plan["arms"]
        ordered = case.get("arms", arms[case_index % len(arms):] + arms[:case_index % len(arms)])
        jobs.extend((case, arm) for arm in ordered)

    def train(job):
        case, arm = job
        workspace = output / case["id"] / arm
        old_workspace = previous_study / case['id'] / arm if previous_study else None
        if old_workspace and (old_workspace / 'training.json').is_file():
            result = json.loads((old_workspace / 'training.json').read_text())
            return {'case': case, 'arm': arm, 'result': result, 'initial_event_count': 0}
        agent = agent_factory(provider_factory(workspace), workspace,
            arm=arm, knowledge=deepcopy(knowledge or {}), config=limits,
            confirmation_seed=case["confirmation_seed"], source_task=not bool(knowledge))
        laboratory = laboratory_factory(case)
        prefix_account = {}
        if old_workspace and old_workspace.is_dir():
            previous = json.loads((old_workspace / 'active.json').read_text())
            agent.restore_research(previous, old_workspace, laboratory,
                                   confirmation_seed=case['confirmation_seed'])
            agent.common_interface += resume_task
            recovery = restore_pending_question_error(agent, laboratory, previous,
                old_workspace, model=plan['model'])
            prefix_account = {'prior_research_cost': deepcopy(agent.costs),
                              'resume_origin': str(old_workspace / 'active.json'),
                              'recovered_paid_reply': recovery}
            write_json(workspace / 'active.json', agent.result('researching'))
        elif prepare_prefix is None:
            prefix = laboratory.confirmation_actions(plan["prefix_rows"], Random(case["prefix_seed"]))
            feedback = agent._execute("query", {"actions": prefix}, laboratory)
            if feedback.get("error"):
                raise ValueError(feedback["error"])
            agent.events.append({"step": -1, "action": "query", "arguments": {"actions": prefix},
                "feedback": feedback, "usage": {"backend": "common_paid_prefix", "total_tokens": 0}})
        else:
            prefix_account = prepare_prefix(agent, laboratory, case)
        initial_event_count = len(agent.events)
        result = agent.run(laboratory, on_checkpoint=lambda data, path=workspace:
                           write_json(path / "active.json", data))
        if prefix_account:
            result.update(prefix_account)
            if "prior_research_cost" in prefix_account:
                prior = prefix_account["prior_research_cost"]
                result["continuation_cost"] = {key: value - prior.get(key, 0)
                    if value is not None else None for key, value in result["costs"].items()}
            else:
                historical = prefix_account["historical_prefix_cost"]
                result["additional_cost"] = {key: value - historical.get(key, 0)
                    if value is not None else None for key, value in result["costs"].items()}
        write_json(workspace / "training.json", result)
        write_json(workspace / "archive.json", result["outgoing_knowledge"])
        print(json.dumps({"case": case["id"], "arm": arm, "status": result["status"],
                          "costs": result["costs"]}), flush=True)

        return {"case": case, "arm": arm, "result": result,
                "initial_event_count": initial_event_count}

    if workers == 1:
        trials = [train(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # map retains registration order; all training finishes before any
            # terminal rows are generated by the separate evaluation loop.
            trials = list(executor.map(train, jobs))

    records = []
    for trial in trials:
        case, result = trial["case"], trial["result"]
        evaluation = {"status": "no_submitted_model", "mse": None}
        if result["model"]:
            rows = laboratory_factory(case).evaluation_rows(
                length=plan["evaluation"]["rows"], seed=case["evaluation_seed"])
            evaluation = evaluate_python_model(result["workspace"], result["model"], rows,
                                               timeout_seconds=limits.execution_seconds)
            evaluation.pop("predictions", None)
        evaluation["passed"] = (evaluation.get("returncode") == 0 and
            evaluation.get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"])
        host_workspace = output / case['id'] / trial['arm']
        host_workspace.mkdir(parents=True, exist_ok=True)
        write_json(host_workspace / "evaluation.json", evaluation)
        usage = [event.get("usage", {}) for event in result["events"]
                 if event.get("usage", {}).get("backend") == "codex_cli"]
        records.append({"case": case["id"], "kind": case["kind"], "arm": trial["arm"],
            "status": result["status"], "model": result["model"], "costs": result["costs"],
            "fresh_confirmation_passed": (result.get("fresh_confirmation") or {}).get("passed", False),
            "terminal": evaluation,
            "revision_count": len(result.get("revision_proposals", [])),
            "cached_input_tokens": sum(item.get("cached_input_tokens") or 0 for item in usage),
            "calls_with_cache_hits": sum(bool(item.get("cached_input_tokens")) for item in usage),
            "archive": str(Path(result["workspace"]) / "archive.json")})
        for field in ("historical_prefix_cost", "additional_cost", "prefix_origin",
                      "prior_research_cost", "continuation_cost", "resume_origin",
                      "recovered_paid_reply"):
            if field in result:
                records[-1][field] = deepcopy(result[field])
    live_events = [event for trial in trials
                   for event in trial["result"]["events"][trial["initial_event_count"]:]
                   if event.get("usage", {}).get("backend") not in
                       {"common_paid_prefix", "restored_paid_prefix"}]
    summary = {"study": plan["study"], "status": "completed", "trials": records,
               "evidence_kind": "real_external_model" if live_events and all(
                   event.get("usage", {}).get("backend") == "codex_cli" for event in live_events)
                   else "scripted_interface_check",
               "source_count": len(plan["cases"]), "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=("memoryless", "delay", "hysteresis"),
                        default=["memoryless", "delay", "hysteresis"])
    parser.add_argument("--cases", type=int, default=1, help="Independent source seeds per control family")
    parser.add_argument("--arms", nargs="+", choices=("code_react", "hegel"), default=["code_react", "hegel"])
    parser.add_argument("--archive", type=Path, help="Identical incoming executable archive for all arms")
    parser.add_argument("--seed", type=int, default=61011017)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--timeout-seconds", type=float, default=300.)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    plan = protocol(kinds=args.kinds, cases=args.cases, arms=args.arms, seed=args.seed,
                    model=args.model, timeout_seconds=args.timeout_seconds, archive=args.archive)
    plan["provider"]["auth_mode"] = verify_codex_subscription()
    knowledge = json.loads(args.archive.read_text()) if args.archive else None
    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model, reasoning_effort="medium",
                                 timeout_seconds=args.timeout_seconds)
    print(json.dumps(run_study(plan, args.output, provider_factory=provider, knowledge=knowledge), indent=2))


if __name__ == "__main__":
    main()
