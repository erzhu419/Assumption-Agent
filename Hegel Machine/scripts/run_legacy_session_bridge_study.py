#!/usr/bin/env python3
"""Measure the existing native revisit loop through the paid inquiry interface."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.legacy_research_bridge import LEGACY_KNOWLEDGE_PATH
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent

COSTS = ("observation_cost", "measurement_cost", "compute_cost")
ARMS = ("retained", "removed")


def protocol():
    return {"study": "legacy_inquiry_session_knowledge_availability",
        "development": {"apparatus_seed": 910061, "seed": 910071, "consumer_channels": ["meter"]},
        "cases": [{"apparatus_seed": 920071 + 100003 * i, "seed": 930071 + 100003 * i,
                   "consumer_channels": []} for i in range(12)],
        "limits": asdict(InquiryLLMConfig(observation_budget=384, compute_budget=360000,
            execution_seconds=120., confirmation_rows=24)),
        "research_config": asdict(ResearchConfig(observation_budget=384, compute_budget=360000,
            minimum_improvement=1e-7, max_rounds=4, max_history=4, active_probes=True, seed=71)),
        "intervention": "Only the complete KnowledgeLibrary differs: the measured producer library "
            "versus its empty pre-production state. Both arms inherit identical paid consumer and "
            "producer reports, observations, completed source research and resource limits. Fork "
            "before the native scheduler proposes a revisit; no queued candidate is supplied.",
        "fixture": "Existing TemporalInquiryLab task_index=1, finite2. Consumer buys three 24-row "
            "trajectories; source invokes the already implemented legacy_research operator. "
            "The development case includes meter readings; cohort consumer evidence uses controls "
            "only, matching the public predictor contract. The source may purchase measurements.",
        "execution": "Controlled offline invocation of the current LLM tools and native session. "
            "There is no external LLM or supplied final law; this is a connection study, not autonomous "
            "LLM action-selection evidence. Native session decides whether and how to revisit.",
        "accounting": "Include the physically shared source production and consumer acquisition once "
            "in each counterfactual arm. Add actual screening, revisit and Agent finish charges. "
            "Each episode retains its original cap and fresh-confirmation reserve.",
        "evaluation": {"rows": 128, "seed_offset": 1000003, "mse_threshold": 1e-10},
        "evaluation_order": "Train every arm in every registered cohort case before any Agent finish "
            "confirmation or terminal evaluation. Finish failures are retained without retraining. "
            "Evaluate each exported frozen model even if Agent confirmation failed.",
        "scope": "Knowledge-triggered continuation versus no available transfer knowledge. "
            "The removed arm is the same native scheduler, not a new ordinary-agent baseline. "
            "Source return means an actual fresh paired comparison, not method support count.",
        "external_model_calls": 0}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_episode(path, case, plan, role):
    seed = case["seed"] + (0 if role == "consumer" else 1)
    agent = TemporalOperatorAgent(None, path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(**plan["limits"]), confirmation_seed=seed + 7919)
    lab = TemporalInquiryLab(task_index=1, seed=seed,
        apparatus_seed=case["apparatus_seed"], instrument_mode="finite2")
    return {"agent": agent, "laboratory": lab, "research": None,
            "available_concepts": [], "seed": seed}


def clone_episode(episode, path):
    """Restore paid data and physical state without buying the prefix again."""
    copied = deepcopy(episode)
    agent = copied["agent"]
    agent.workspace = Path(path).resolve()
    agent.workspace.mkdir(parents=True, exist_ok=True)
    for name, contents in agent.files.items():
        target = agent.workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    agent._write_evidence()
    return copied


def summed_costs(episodes):
    return {key: sum(episode["agent"].costs[key] for episode in episodes) for key in COSTS}


def train_case(case, output, plan):
    from hegel_machine.cognition.legacy_session_bridge import LegacyInquirySession
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "case_protocol.json", case)
    consumer = make_episode(output / "common" / "consumer", case, plan, "consumer")
    rng = Random(case["seed"])
    consumer["agent"]._execute("query", {"actions": [{"kind": "trajectory",
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)],
        "channels": list(case["consumer_channels"])} for _ in range(3)]}, consumer["laboratory"])
    source = make_episode(output / "common" / "source", case, plan, "source")
    produced = source["agent"]._execute("run", {"operator": "legacy_research"}, source["laboratory"])
    source["research"] = json.loads((source["agent"].workspace / produced["report_path"]).read_text())
    library = json.loads((source["agent"].workspace / LEGACY_KNOWLEDGE_PATH).read_text())
    shared = [consumer, source]
    common_costs = summed_costs(shared)
    write_json(output / "common.json", {"costs": common_costs, "production": produced,
        "knowledge": library, "episodes": [episode["agent"].result("prepared") for episode in shared]})
    arms = {}
    for arm in ARMS:
        episodes = [clone_episode(episode, output / arm / role)
                    for episode, role in zip(shared, ("consumer", "source"))]
        knowledge = deepcopy(library) if arm == "retained" else KnowledgeLibrary().to_dict()
        session = LegacyInquirySession(episodes, knowledge=knowledge,
            config=ResearchConfig(**plan["research_config"]), controller_position=0)
        initial = deepcopy(session.session.to_dict())
        if initial["pending_revisits"] or initial["evaluation"]:
            raise ValueError("the intervention must precede revisit planning and terminal scoring")
        write_json(output / arm / "initial.json", initial)
        trained = session.run()
        write_json(output / arm / "training.json", trained)
        arms[arm] = {"episodes": episodes, "state": trained, "initial": initial, "bridge": session}
    return {"case": deepcopy(case), "output": output, "common_costs": common_costs,
            "source_added_concepts": produced.get("concepts_added", []), "arms": arms}


def score_case(trained, plan):
    results = {}
    for arm, payload in trained["arms"].items():
        state = payload["state"]
        return_confirmations = []
        if plan.get("source_confirmation_action"):
            for position, episode in enumerate(payload["episodes"]):
                feedback = state["results"][position]["research"].get("knowledge_return") or {}
                provisional = [row for row in feedback.get("returns", [])
                               if row.get("confirmation_status") != "confirmed"]
                if provisional:
                    result = episode["agent"]._execute("run",
                        {"operator": plan["source_confirmation_action"], "length": 24}, episode["laboratory"])
                    return_confirmations.append({"task_position": position, "result": result})
        confirmations = []
        for position, episode in enumerate(payload["episodes"]):
            exported = next(item for item in state["inquiry_exports"] if item["task_position"] == position)
            model = exported.get("model_path")
            confirmations.append(episode["agent"]._execute("finish", {"model": model}, episode["laboratory"])
                if model else {"error": exported.get("export_reason", "no exported model")})
        # Agent finish may update actual confirmation and source-return status.
        # Retain the pre-finish training snapshot as well as the final native state.
        state = payload["bridge"].to_dict()
        write_json(trained["output"] / arm / "confirmed_session.json", state)
        tasks = []
        for position, episode in enumerate(payload["episodes"]):
            agent, lab = episode["agent"], episode["laboratory"]
            exported = next(item for item in state["inquiry_exports"] if item["task_position"] == position)
            model = exported.get("model_path")
            confirmation = confirmations[position]
            rows = lab.evaluation_rows(length=plan["evaluation"]["rows"],
                seed=episode["seed"] + plan["evaluation"]["seed_offset"])
            terminal = (evaluate_python_model(agent.workspace, model, rows) if model
                        else {"returncode": None, "error": "no exported model"})
            terminal.pop("predictions", None)
            fresh = confirmation.get("fresh_confirmation", {}).get("passed", False)
            correct = terminal.get("returncode") == 0 and terminal.get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"]
            research = state["results"][position]["research"]
            tasks.append({"task_position": position, "role": "consumer" if position == 0 else "source",
                "research_status": research["status"], "native_confirmation_loss": research.get("confirmation_loss"),
                "selected_channels": research.get("selected_channels", []),
                "model": model, "export_reason": exported.get("export_reason"),
                "fresh_confirmation_passed": fresh, "terminal_correct": correct,
                "confirmed_and_correct": fresh and correct, "terminal": terminal,
                "confirmation": confirmation, "costs": deepcopy(agent.costs),
                "knowledge_return": deepcopy(research.get("knowledge_return")),
                "used_sources": research.get("used_sources", [])})
            write_json(agent.workspace / "final.json", agent.result("finished" if fresh else "not_confirmed"))
        records = [receipt for task in tasks for receipt in (task["knowledge_return"] or {}).get("returns", [])]
        outcomes = [{"method_id": method["method_id"], **deepcopy(outcome)}
            for method in state["knowledge"]["research_workflows"]["methods"]
            for outcome in method.get("knowledge_outcomes", [])]
        costs = summed_costs(payload["episodes"])
        results[arm] = {"tasks": tasks, "costs": costs,
            "incremental_costs": {key: costs[key] - trained["common_costs"][key] for key in COSTS},
            "revisits": deepcopy(state["revisits"]), "scheduler_events": deepcopy(state["scheduler_events"]),
            "source_returns": records, "confirmed_positive_source_returns": sum(
                row.get("confirmation_status") == "confirmed" and row.get("retained_normalized_gain", 0) > 0
                for row in records), "workflow_knowledge_outcomes": outcomes,
            "source_confirmation_statuses": dict(Counter(row.get("confirmation_status", "unspecified") for row in records)),
            "negative_source_returns": sum(row.get("retained_normalized_gain", 0) < 0 for row in records),
            "measurement_required_export_failures": sum(task["model"] is None and bool(task["selected_channels"])
                for task in tasks),
            "export_failures_by_reason": dict(Counter(task["export_reason"] for task in tasks if task["model"] is None)),
            "source_confirmation_actions": return_confirmations,
            "external_model_calls": sum(episode["agent"].costs["model_calls"] for episode in payload["episodes"])}
    result = {"case": trained["case"], "common_costs": trained["common_costs"],
        "source_added_concepts": trained["source_added_concepts"], "results": results,
        "paired_retained_minus_removed": {key: results["retained"]["costs"][key] - results["removed"]["costs"][key]
            for key in COSTS}}
    write_json(trained["output"] / "results.json", result)
    return result


def summarize(records, failures, registered):
    arms = {}
    for arm in ARMS:
        results = [record["results"][arm] for record in records]
        statuses, export_failures = Counter(), Counter()
        for result in results:
            statuses.update(result.get("source_confirmation_statuses", {}))
            export_failures.update(result.get("export_failures_by_reason", {}))
        arms[arm] = {"consumer_confirmed_and_correct": sum(result["tasks"][0]["confirmed_and_correct"] for result in results),
            "source_confirmed_and_correct": sum(result["tasks"][1]["confirmed_and_correct"] for result in results),
            "costs": {key: sum(result["costs"][key] for result in results) for key in COSTS},
            "incremental_costs": {key: sum(result["incremental_costs"][key] for result in results) for key in COSTS},
            "revisits": sum(len(result["revisits"]) for result in results),
            "confirmed_positive_source_returns": sum(result["confirmed_positive_source_returns"] for result in results),
            "source_confirmation_statuses": dict(statuses),
            "negative_source_returns": sum(result.get("negative_source_returns", 0) for result in results),
            "measurement_required_export_failures": sum(result.get("measurement_required_export_failures", 0) for result in results),
            "export_failures_by_reason": dict(export_failures),
            "workflow_knowledge_outcomes": sum(len(result["workflow_knowledge_outcomes"]) for result in results),
            "source_confirmation_actions": sum(len(result.get("source_confirmation_actions", [])) for result in results),
            "external_model_calls": sum(result["external_model_calls"] for result in results)}
    return {"registered_cases": registered, "completed_cases": len(records), "execution_failures": failures,
            "arms": arms, "paired_retained_minus_removed": {key: arms["retained"]["costs"][key] - arms["removed"]["costs"][key]
                for key in COSTS}, "external_model_calls": sum(arm["external_model_calls"] for arm in arms.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "development_return", "cohort"), default="development")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registration = args.output / "protocol.json"
    if registration.exists():
        plan = json.loads(registration.read_text())
    else:
        plan = protocol()
        write_json(registration, plan)
    if args.phase == "development_return":
        plan = deepcopy(plan)
        plan["source_confirmation_action"] = "legacy_return_confirmation"
        plan["return_confirmation_scope"] = (
            "One public paid source-return confirmation per task with an existing unconfirmed "
            "return after native learning. Preserve the earlier meter development result; "
            "this separately registered continuation never changes the fixed cohort protocol.")
        write_json(args.output / args.phase / "protocol.json", plan)
    cases = plan["cases"] if args.phase == "cohort" else [plan["development"]]
    trained, failures = [], []
    for case in cases:
        path = args.output / args.phase / f"case_{case['apparatus_seed']}"
        try:
            trained.append(train_case(case, path, plan))
            print(json.dumps({"phase": "trained", "apparatus_seed": case["apparatus_seed"]}), flush=True)
        except Exception:
            failure = {"case": case, "phase": "training", "error": traceback.format_exc()}
            failures.append(failure)
            write_json(path / "failure.json", failure)
            print(json.dumps({"phase": "training_failed", "apparatus_seed": case["apparatus_seed"]}), flush=True)
    # No Agent finish or terminal labels exist while another arm is still learning.
    records = []
    for item in trained:
        try:
            records.append(score_case(item, plan))
        except Exception:
            failure = {"case": item["case"], "phase": "scoring", "error": traceback.format_exc()}
            failures.append(failure)
            write_json(item["output"] / "failure.json", failure)
    summary = summarize(records, failures, len(cases))
    write_json(args.output / args.phase / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
