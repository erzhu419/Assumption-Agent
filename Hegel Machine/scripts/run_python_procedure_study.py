#!/usr/bin/env python3
"""Development replay of atomic research versus an explicitly chosen procedure."""

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
from hegel_machine.cognition.temporal_paired_method import (
    DEFAULT_RESEARCH_PROCEDURE, FROZEN_METHOD_INTERFACE, PROCEDURE_INTERFACE,
    ProcedureConsumerAgent,
)
from scripts.run_candidate_retention_cycle import task_for
from scripts.run_candidate_retention_cycle_llm import (
    execute_episode, prepare_episode, score_episode, write_json,
)
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription
from scripts.run_python_pending_advice_study import protocol as pending_protocol

SOURCE_OUTCOMES = ROOT / "artifacts/python_pending_advice_luna_20261006/source_outcomes.json"
PROGRAM_PATH = "shared/research_procedure.py"
SHARED_PROGRAM_INTERFACE = """
An editable research procedure is supplied at shared/research_procedure.py.
You may inspect, change, or invoke it with query {path:'shared/research_procedure.py'}.
This query invokes select_actions once using current paid evidence. You choose
whether to use it, repeat it, or take ordinary research actions yourself.
""" + FROZEN_METHOD_INTERFACE
USAGE_KEYS = ("known_tokens", "input_tokens", "cached_input_tokens", "missing_usage_calls")


class AtomicProcedureAgent(ProcedureConsumerAgent):
    """Keep the same program and single-step contract, without bundled execution."""

    common_interface = ProcedureConsumerAgent.common_interface.removesuffix(PROCEDURE_INTERFACE)

    def _execute(self, action, arguments, laboratory):
        if action == "run" and arguments.get("operator") == "procedure":
            raise ValueError("procedure is unavailable in the atomic condition; query a program once")
        return super()._execute(action, arguments, laboratory)


def protocol(mode, sources):
    plan = pending_protocol()
    plan.update(study="python_explicit_procedure_development_replay", mode=mode,
        conditions=["bundled", "atomic"] if mode == "live" else ["bundled"],
        cases=[deepcopy(source["case"]) for source in sources if source["candidate"] is not None],
        source=str(SOURCE_OUTCOMES), procedure_step_limit=12,
        intervention="Both conditions receive the identical editable, developer-authored "
            "procedure, persistent advice, single-step query contract and numerical tools. "
            "Only bundled exposes the procedure operator for explicitly bounded repeated "
            "execution; confirm:true explicitly permits its ordinary fresh confirmation. "
            "The supplied adapter reads public research advice or invokes research_choice.",
        controller=("After the common paid prefix, Luna chooses every action, including whether "
            "to invoke or edit the procedure and whether to authorize confirmation. Maximum "
            "12 model calls; internal procedure steps consume the same observation, compute "
            "and execution budgets but are not model calls." if mode == "live" else
            "Offline integration replay: invoke the supplied procedure once per target with "
            "max_steps:12 and confirm:true. No LLM and no atomic condition are run. This checks "
            "the chosen procedure's execution, not autonomous adoption or a paired advantage."),
        accounting="No source is relearned. Preserve all four prior acquisition outcomes and "
            "their costs separately, including the failed source; report the three reused "
            "sources' costs as a subset. Actual new costs contain only this run's targets. "
            "Count actual operator executions, program steps and provider calls separately.",
        failure_policy="This development replay is conditional on the three existing published "
            "archives. The earlier failed source is disclosed, not replaced or retried. Keep "
            "every registered target failure and service interruption; unknown usage stays null.",
        terminal_order="Finish all target learning before any terminal scoring. Existing source "
            "publication, not prior target or terminal correctness, determines archive admission.",
        scope="Reuse three actual Luna source archives and the same six development targets "
            "and paid-prefix seeds from the preceding persistence study. Live has six paired "
            "targets and twelve episodes; offline has six bundled episodes only. This tests "
            "explicit procedure execution availability, not new task generalization, learned "
            "method discovery or a vanilla-versus-Hegel comparison.")
    plan.pop("adoption", None)
    if mode == "offline":
        plan["model"] = None
        plan["limits"].update(max_model_calls=0, token_budget=0)
    return plan


def target_jobs(sources, output, plan):
    jobs = []
    for index, source in enumerate(sources):
        if source["candidate"] is None:
            continue
        for kind_index, kind in enumerate(plan["target_kinds"]):
            order = plan["conditions"] if (index + kind_index) % 2 == 0 else plan["conditions"][::-1]
            for condition in order:
                jobs.append({"case": source["case"], "kind": kind, "condition": condition,
                    "candidate": source["candidate"], "path": Path(output) / f"trial_{len(jobs):03d}"})
    return jobs


def prepare_target(job, plan, provider=None, *, agent_class=None):
    case = job["case"]
    settings = task_for(case, 1)
    if job["kind"] == "recurrent_instrument_change":
        settings.update(instrument_mode="recurrent", response_seed=case["apparatus_seed"])
    payload = prepare_episode(case, 1, job["path"], [deepcopy(job["candidate"])], plan,
        provider=provider, laboratory=TemporalInquiryLab(**settings),
        agent_class=agent_class or (ProcedureConsumerAgent if job["condition"] == "bundled" else AtomicProcedureAgent))
    agent = payload["agent"]
    agent.common_interface += SHARED_PROGRAM_INTERFACE
    if job["condition"] == "bundled":
        agent.common_interface += PROCEDURE_INTERFACE
    agent._save(PROGRAM_PATH, DEFAULT_RESEARCH_PROCEDURE)
    agent.knowledge["files"] = deepcopy(agent.files)
    agent.inherited_files = set(agent.files)
    return payload


def offline_research(payload, max_steps):
    agent, started = payload["agent"], monotonic()
    arguments = {"operator": "procedure", "path": PROGRAM_PATH,
                 "max_steps": max_steps, "confirm": True}
    status = "procedure_not_completed"
    try:
        feedback = agent._execute("run", arguments, payload["laboratory"])
        status = "completed" if agent.model else feedback.get("status", status)
    except Exception:
        status = "execution_failed"
        feedback = {"error": traceback.format_exc()}
    agent.events.append({"step": len(agent.events), "action": "run", "arguments": arguments,
        "feedback": deepcopy(feedback), "usage": {"backend": "offline_procedure_execution",
        "total_tokens": 0, "input_tokens": 0, "cached_input_tokens": 0}})
    payload["research_wall_seconds"] = monotonic() - started
    payload["training"] = agent.result(status)
    write_json(payload["path"] / "training.json", payload["training"])
    return payload


def execution_evidence(agent, training):
    """Use execution ledgers once; feedback can contain duplicate nested traces."""
    operators = deepcopy(agent.operator_results)
    methods = deepcopy(agent.method_results)
    return {"operator_calls": dict(Counter(item["operator"] for item in operators)),
        "executed_operators": operators, "method_steps": len(methods), "method_results": methods,
        "procedure_requests": sum(event.get("action") == "run" and
            event.get("arguments", {}).get("operator") == "procedure" for event in training["events"]),
        "program_action_counts": dict(Counter(
            item["proposal"].get("action", "finish" if "model" in item["proposal"] else "query")
            for item in methods if isinstance(item.get("proposal"), dict)))}


def summarize(sources, results, plan):
    def aggregate(items):
        return {"attempted_targets": len(items),
            "confirmed_and_correct": sum(item["confirmed_and_correct"] for item in items),
            "false_confirmations": sum(item["false_confirmation"] for item in items),
            "statuses": dict(Counter(item["status"] for item in items)),
            "costs": total_costs([item["costs"] for item in items]),
            **{key: sum(item[key] for item in items) for key in USAGE_KEYS},
            "procedure_requests": sum(item["procedure_requests"] for item in items),
            "method_steps": sum(item["method_steps"] for item in items),
            "operator_calls": dict(sum((Counter(item["operator_calls"]) for item in items), Counter()))}

    conditions = {}
    for condition in plan["conditions"]:
        selected = [item for item in results if item["condition"] == condition]
        conditions[condition] = {"registered_targets": len(plan["cases"]) * len(plan["target_kinds"]),
            **aggregate(selected), "by_kind": {kind: aggregate([item for item in selected if item["kind"] == kind])
                                               for kind in plan["target_kinds"]}}
    return {"study": plan["study"], "mode": plan["mode"], "conditions": conditions,
        "registered_episodes": len(plan["cases"]) * len(plan["target_kinds"]) * len(plan["conditions"]),
        "existing_sources": sources,
        "existing_all_source_acquisition_costs": total_costs([item["costs"] for item in sources]),
        "existing_reused_source_costs": total_costs([item["costs"] for item in sources if item["candidate"] is not None]),
        "prior_sources_without_published_candidate": sum(item["candidate"] is None for item in sources),
        "actual_new_targets": aggregate(results), "results": results, "scope": plan["scope"]}


def run_study(output, *, mode="offline", provider_factory=None, auth_mode=None):
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    plan = protocol(mode, sources)
    plan["provider"]["auth_mode"] = auth_mode
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_python_pending_advice_study.py", "run_python_advice_ablation.py",
                 "run_python_research_routing.py", "run_python_refit_diagnosis.py", "run_candidate_retention_cycle_llm.py",
                 "run_candidate_retention_cycle.py", "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    jobs = target_jobs(sources, output, plan)

    def train(job):
        payload = prepare_target(job, plan, provider_factory(job["path"]) if provider_factory else None)
        return job, (execute_episode(payload) if mode == "live" else
                     offline_research(payload, plan["procedure_step_limit"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, jobs))
    results = []
    # Every training episode has stopped before any private terminal labels exist.
    for job, payload in trained:
        result = score_episode(payload, plan)
        result.update(case=job["case"], kind=job["kind"], condition=job["condition"], workspace=str(payload["path"]),
                      **execution_evidence(payload["agent"], payload["training"]))
        write_json(output / "evaluations" / (payload["path"].name + ".json"), result)
        results.append(result)
    summary = summarize(sources, results, plan)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = {}
    if args.mode == "live":
        kwargs = {"auth_mode": verify_codex_subscription(), "provider_factory": lambda path:
            CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180)}
    summary = run_study(args.output, mode=args.mode, **kwargs)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"existing_sources", "results"}}, indent=2))


if __name__ == "__main__":
    main()
