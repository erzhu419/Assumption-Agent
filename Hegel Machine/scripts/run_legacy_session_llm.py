#!/usr/bin/env python3
"""Let the same real LLM continue a paid session with or without prior knowledge."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider, evaluate_python_model
from hegel_machine.cognition.legacy_research_bridge import LEGACY_KNOWLEDGE_PATH
from hegel_machine.cognition.legacy_session_bridge import LegacyInquirySession
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.online_inquiry_llm import COMMON_INTERFACE, InquiryLLMConfig
from hegel_machine.cognition.temporal_operator_agent import OPERATOR_INTERFACE
from scripts import run_legacy_session_bridge_study as paid_study
from scripts.run_online_inquiry_llm import verify_codex_subscription


CONDITIONS = ("retained", "task_local")
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost",
             "python_execution_seconds", "model_calls", "reported_tokens")
TASK_INTERFACE = """
You are continuing the consumer task, position 0, in this paid research session.
Three control trajectories have already been observed. Their records and costs
are available in the ordinary evidence files. The current zero predictor is an
unassessed starting point, not an accepted hypothesis. Another task was studied
by the shared native researcher before this continuation. The available library
is in shared/legacy_knowledge.json; it may be empty. Inspect, use or revise any
available knowledge when the current evidence supports it. You may also research
from the current observations using ordinary Python and the shared tools.
Infer an accurate controls-only predictor for the current apparatus and submit
it with finish. Select your own research actions; no operator sequence is required.
The native research report and a completed native status do not submit a model.
No knowledge manifest is required for this continuation.
"""


def write_json(path, value):
    write_json_checkpoint(path, value, indent=2, ensure_ascii=True, allow_nan=False)


def protocol():
    original = paid_study.protocol()
    return {
        "study": "legacy_session_same_llm_continuation", "model": "gpt-6-luna",
        "provider": {"backend": "codex_cli", "reasoning_effort": "medium",
                     "timeout_seconds": 180},
        "cases": deepcopy(original["cases"][:4]), "conditions": list(CONDITIONS),
        "limits": asdict(InquiryLLMConfig(observation_budget=384, compute_budget=360000,
            execution_seconds=120., max_model_calls=12, token_budget=400000,
            confirmation_rows=24)),
        "research_config": deepcopy(original["research_config"]),
        "evaluation": deepcopy(original["evaluation"]),
        "intervention": "Only availability of the complete measured producer KnowledgeLibrary "
            "differs. Both conditions have the same paid consumer prefix, completed producer, "
            "model, tools, per-episode limits and confirmation seed. Fork before revisit planning.",
        "source": "Reproduce the existing paid preparation using its make_episode/clone_episode "
            "helpers and public tools: three 24-row consumer trajectories and native source research. "
            "The producer is an existing offline algorithm, not an LLM-authored source discovery.",
        "execution": "The host binds a session but never calls its run or Agent finish. The real "
            "LLM selects every continuation action, including ordinary Python, native research, "
            "revisit and final submission. Both conditions may learn new knowledge.",
        "accounting": "Include source production once in each counterfactual condition; report "
            "source costs, consumer-prefix costs, consumer continuation costs and session totals. "
            "The original source was physically produced once per apparatus before forking.",
        "evaluation_order": "Stop all eight LLM continuations before generating any 128-row "
            "terminal evaluation. Only LLM-selected finish invokes fresh confirmation during research.",
        "failure_policy": "Keep every attempt, including provider, parsing, execution, research "
            "and budget failure. No replacement run or automatic provider retry.",
        "scope": "Four previously studied apparatuses, not a new independent apparatus cohort. "
            "This tests LLM control of a joined session after a common paid prefix. task_local has "
            "all native tools and is not a tool-restricted vanilla Code/ReAct baseline. A final "
            "source return is counted only for the submitted frozen session predictor. Selecting "
            "finish can trigger automatic source settlement; explicit return-confirmation choice "
            "is reported separately.",
        "model_identity": "gpt-6-luna is explicitly requested. Provider usage may repeat the "
            "requested model; this is not independent evidence of server-side model routing.",
    }


def total_costs(items):
    return {key: (sum(item.get(key, 0) for item in items)
                  if all(isinstance(item.get(key, 0), (int, float)) for item in items)
                  else None) for key in COST_KEYS}


def prepare_case(case, path, plan):
    """Reproduce the already measured paid prefix; make no LLM decisions."""
    path = Path(path)
    consumer = paid_study.make_episode(path / "consumer", case, plan, "consumer")
    rng = Random(case["seed"])
    consumer["agent"]._execute("query", {"actions": [{"kind": "trajectory",
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)],
        "channels": list(case["consumer_channels"])} for _ in range(3)]}, consumer["laboratory"])
    source = paid_study.make_episode(path / "source", case, plan, "source")
    production = source["agent"]._execute("run", {"operator": "legacy_research"}, source["laboratory"])
    source["research"] = json.loads((source["agent"].workspace / production["report_path"]).read_text())
    library = json.loads((source["agent"].workspace / LEGACY_KNOWLEDGE_PATH).read_text())
    setup = {"consumer_prefix": deepcopy(consumer["agent"].costs),
             "source": deepcopy(source["agent"].costs)}
    write_json(path / "common.json", {"case": case, "production": production,
        "knowledge": library, "setup_costs": setup,
        "episodes": [item["agent"].result("prepared") for item in (consumer, source)]})
    return {"case": deepcopy(case), "episodes": [consumer, source],
            "knowledge": library, "setup_costs": setup}


def prepare_arm(common, condition, path, plan, provider_factory):
    path = Path(path)
    episodes = [paid_study.clone_episode(episode, path / role)
        for episode, role in zip(common["episodes"], ("consumer", "source"))]
    agent = episodes[0]["agent"]
    agent.provider = provider_factory(agent.workspace)
    agent.common_interface = (COMMON_INTERFACE.split("For the hegel condition,")[0]
                              + OPERATOR_INTERFACE + TASK_INTERFACE)
    knowledge = (deepcopy(common["knowledge"]) if condition == "retained"
                 else KnowledgeLibrary().to_dict())
    bridge = LegacyInquirySession(episodes, knowledge=knowledge,
        config=ResearchConfig(**plan["research_config"]), controller_position=0)
    initial = bridge.to_dict()
    if initial["pending_revisits"] or initial["revisits"] or initial["evaluation"]:
        raise ValueError("fork must precede native planning, revisits and terminal evaluation")
    write_json(path / "initial_session.json", initial)
    write_json(path / "preparation.json", {"case": common["case"], "condition": condition,
        "setup_costs": common["setup_costs"], "consumer": agent.result("prepared")})
    return {"case": deepcopy(common["case"]), "condition": condition, "path": path,
            "episodes": episodes, "bridge": bridge,
            "setup_costs": deepcopy(common["setup_costs"])}


def run_episode(payload):
    episode = payload["episodes"][0]
    agent, lab = episode["agent"], episode["laboratory"]

    def checkpoint(result):
        write_json(payload["path"] / "active.json", result)
        event = result["events"][-1] if result["events"] else {}
        print(json.dumps({"apparatus_seed": payload["case"]["apparatus_seed"],
            "condition": payload["condition"], "status": result["status"],
            "calls": result["costs"]["model_calls"], "action": event.get("action"),
            "operator": event.get("arguments", {}).get("operator")}), flush=True)

    try:
        result = agent.run(lab, on_checkpoint=checkpoint)
    except Exception:
        result = agent.result("execution_failed")
        result["execution_error"] = traceback.format_exc()
    payload["training"] = result
    write_json(payload["path"] / "training.json", result)
    write_json(payload["path"] / "trained_session.json", payload["bridge"].to_dict())
    return payload


def final_source_status(payload):
    """Keep native history distinct from attribution of the actual submission."""
    state = payload["bridge"].to_dict()
    agent = payload["episodes"][0]["agent"]
    research = state["results"][0]["research"]
    context = state.get("inquiry_confirmation_contexts", {}).get("0")
    matching = bool(agent.model and context and agent.model == context["model_path"]
        and context["program"] == research["program"]
        and (agent.workspace / agent.model).read_text() == context["code"])
    native = (research.get("knowledge_return") or {}).get("returns", [])
    receipts = native if matching else []
    outcomes = [deepcopy(item) for method in state["knowledge"]["research_workflows"]["methods"]
        for item in method.get("knowledge_outcomes", [])
        if item.get("downstream_task") == research["task_id"]]
    confirmations = [event.get("feedback", {}).get("source_confirmation")
                    for event in payload["training"]["events"]
                    if event.get("feedback", {}).get("source_confirmation")]
    return {"submitted_source_binding": ("matching_frozen_session_predictor" if matching
        else "no_submitted_model" if not agent.model else "different_submitted_predictor"),
        "native_source_returns": deepcopy(native), "final_source_returns": deepcopy(receipts),
        "confirmed_positive_final_source_returns": sum(row.get("confirmation_status") == "confirmed"
            and row.get("retained_normalized_gain", 0) > 0 for row in receipts),
        "final_return_statuses": dict(Counter(row.get("confirmation_status", "unspecified")
                                             for row in receipts)),
        "native_workflow_outcomes": outcomes,
        "finish_source_confirmations": confirmations,
        "finish_source_statuses": dict(Counter(item["status"] for item in confirmations)),
        "finish_source_deferred_reasons": dict(Counter(item.get("reason", "unspecified")
            for item in confirmations if item["status"] == "deferred")),
        "session_revisits": deepcopy(state["revisits"]),
        "explicit_source_confirmations": deepcopy(state.get("inquiry_source_confirmations", []))}


def score_episode(payload, plan):
    result = payload["training"]
    episode = payload["episodes"][0]
    agent, lab = episode["agent"], episode["laboratory"]
    attribution = final_source_status(payload)
    rows = lab.evaluation_rows(length=plan["evaluation"]["rows"],
        seed=episode["seed"] + plan["evaluation"]["seed_offset"])
    terminal = (evaluate_python_model(agent.workspace, result["model"], rows,
                timeout_seconds=plan["limits"]["execution_seconds"]) if result.get("model")
                else {"status": "no_submitted_model"})
    terminal.pop("predictions", None)
    terminal["rows"] = len(rows)
    confirmed = bool((result.get("fresh_confirmation") or {}).get("passed"))
    correct = terminal.get("returncode") == 0 and terminal.get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"]
    events = result["events"]
    usage = [event.get("usage", {}) for event in events]
    consumer_costs = result["costs"]
    prefix = payload["setup_costs"]["consumer_prefix"]
    total = total_costs([item["agent"].costs for item in payload["episodes"]])
    summary = {"apparatus_seed": payload["case"]["apparatus_seed"],
        "condition": payload["condition"], "status": result["status"], "model": result.get("model"),
        "fresh_confirmed": confirmed, "terminal_correct": correct,
        "confirmed_and_correct": confirmed and correct, "false_confirmation": confirmed and not correct,
        "terminal": terminal, "setup_costs": deepcopy(payload["setup_costs"]),
        "consumer_costs": deepcopy(consumer_costs),
        "consumer_continuation_costs": {key: consumer_costs[key] - prefix[key]
            if isinstance(consumer_costs[key], (int, float)) else None for key in COST_KEYS},
        "session_costs": total,
        "llm_actions": dict(Counter(event.get("action", "unknown") for event in events)),
        "llm_operator_choices": dict(Counter(event.get("arguments", {})["operator"]
            for event in events if event.get("action") == "run" and "operator" in event.get("arguments", {}))),
        "executed_operators": deepcopy(agent.operator_results),
        "input_tokens": sum(item.get("input_tokens") or 0 for item in usage),
        "cached_input_tokens": sum(item.get("cached_input_tokens") or 0 for item in usage),
        "cache_usage_reported_calls": sum(isinstance(item.get("cached_input_tokens"), int) for item in usage),
        "requested_model": plan["model"],
        "usage_model_values": dict(Counter(item.get("model", "unreported") for item in usage)),
        **attribution}
    if result.get("execution_error"):
        summary["execution_error"] = result["execution_error"]
    write_json(payload["path"] / "evaluation.json", summary)
    return summary


def summarize(records, failures, plan):
    conditions = {}
    for condition in plan["conditions"]:
        items = [item for item in records if item["condition"] == condition]
        choices = Counter()
        for item in items:
            choices.update(item["llm_operator_choices"])
        conditions[condition] = {"episodes": len(items),
            "confirmed_and_correct": sum(item["confirmed_and_correct"] for item in items),
            "false_confirmations": sum(item["false_confirmation"] for item in items),
            "statuses": dict(Counter(item["status"] for item in items)),
            "source_preparation_costs": total_costs([item["setup_costs"]["source"] for item in items]),
            "consumer_prefix_costs": total_costs([item["setup_costs"]["consumer_prefix"] for item in items]),
            "consumer_costs": total_costs([item["consumer_costs"] for item in items]),
            "consumer_continuation_costs": total_costs([item["consumer_continuation_costs"] for item in items]),
            "session_costs": total_costs([item["session_costs"] for item in items]),
            "llm_operator_choices": dict(choices),
            "episodes_with_revisit_choice": sum(item["llm_operator_choices"].get("legacy_revisit", 0) > 0 for item in items),
            "episodes_with_confirmed_final_source_return": sum(item["confirmed_positive_final_source_returns"] > 0 for item in items),
            "submitted_source_bindings": dict(Counter(item["submitted_source_binding"] for item in items)),
            "input_tokens": sum(item["input_tokens"] for item in items),
            "cached_input_tokens": sum(item["cached_input_tokens"] for item in items)}
    return {"study": plan["study"], "registered_episodes": len(plan["cases"]) * len(plan["conditions"]),
        "scored_episodes": len(records), "failures": failures, "conditions": conditions,
        "episodes": records, "model_identity": plan["model_identity"], "scope": plan["scope"]}


def run_study(output, *, provider_factory, auth_mode=None, max_workers=2, study_protocol=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory; existing attempts are preserved")
    plan = deepcopy(study_protocol) if study_protocol is not None else protocol()
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    # Preserve the existing researcher too, since both arms may choose it.
    shutil.copytree(ROOT / "src" / "hegel_machine" / "cognition",
        output / "source_at_run" / "src" / "hegel_machine" / "cognition",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run_legacy_session_llm.py", "run_legacy_session_bridge_study.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run" / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    payloads, failures = [], []
    for case in plan["cases"]:
        path = output / f"case_{case['apparatus_seed']}"
        try:
            common = prepare_case(case, path / "common", plan)
            for condition in plan["conditions"]:
                payloads.append(prepare_arm(common, condition, path / condition, plan, provider_factory))
        except Exception:
            failure = {"case": case, "phase": "preparation", "error": traceback.format_exc()}
            failures.append(failure)
            write_json(path / "preparation_failure.json", failure)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        trained = list(pool.map(run_episode, payloads))
    records = []
    for payload in trained:
        try:
            records.append(score_episode(payload, plan))
        except Exception:
            failure = {"case": payload["case"], "condition": payload["condition"],
                       "phase": "evaluation", "error": traceback.format_exc()}
            failures.append(failure)
            write_json(payload["path"] / "evaluation_failure.json", failure)
    summary = summarize(records, failures, plan)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = verify_codex_subscription()
    result = run_study(args.output, auth_mode=auth, provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180))
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
