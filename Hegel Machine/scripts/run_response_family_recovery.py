#!/usr/bin/env python3
"""Continue the two recorded family-refutation cases through the existing inquiry."""

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.consolidate_response_measurement import TransferSignalLab
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json

CASES = ("changed_mechanism", "uninformative_sensor")
DEFAULT_SOURCE = ROOT / "artifacts/response_consequence_priority_20261009"


def declarations(source):
    return next((ast.literal_eval(node.value) for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "ESTIMATORS" for target in node.targets)), [])


def account(agent):
    result = agent.result()
    result["costs"].update(observation_cost=result["costs"]["environment_steps"],
        compute_cost=result["costs"]["research_compute_units"])
    return result


def restore_case(source, output, case, *, calls=32):
    """Restore paid evidence without constructing a provider or making model calls."""
    if case not in CASES:
        raise ValueError(f"Unknown continuation case: {case}")
    source, output = Path(source).resolve(), Path(output).resolve()
    previous_dir = source / case / "consolidated"
    previous = json.loads((previous_dir / "result.json").read_text())
    previous["episode_status"] = previous["status"]
    limits = deepcopy(json.loads((source / "protocol.json").read_text())["limits"])
    limits.update(max_model_calls=previous["costs"]["model_calls"] + calls,
        python_seconds_per_task=previous["costs"]["python_execution_seconds"] + 240,
        task_wall_seconds=previous["costs"]["wall_time_seconds"] + 1800,
        provider_request_timeout_seconds=180)
    output.mkdir(parents=True, exist_ok=False)
    agent = DiscoveryWorldAgent(None, output / "agent", arm="hegel", limits=limits)
    lab = TransferSignalLab(case)
    agent.lab = lab
    replay = restore_repair(agent, lab, previous, previous_dir / "agent")
    original = (agent.workspace / METHOD_PATH).read_text()
    (output / "source_before.py").write_text(original)
    context = {"source": str(previous_dir), "case": case, "replay": replay,
        "costs_before": account(agent)["costs"], "prefix_observations": len(agent.observations),
        "prefix_events": len(agent.events), "declarations_before": declarations(original)}
    write_json(output / "replay.json", replay)
    write_json(output / "protocol.json", {"study": "live_recovery_after_recorded_family_refutation",
        "source": str(previous_dir), "case": case, "limits": limits,
        "additional_model_calls": calls, "procedure_max_steps": 256,
        "model": {"requested_alias": "gpt-6-luna", "reasoning_effort": "medium"},
        "auth_mode": None, "costs_before": context["costs_before"],
        "scope": "Continue an existing development failure with its paid records, research state, "
                 "unchanged program and cumulative account. The original parent chooses the inquiry; "
                 "the model chooses its hypothesis, physical experiments and stopping point. "
                 "No scientific declaration or target setting is supplied by this harness.",
        "fixture_boundary": "The uninformative case has only the public density, gap and signal "
                            "instruments; signal repeats density. No additional sensor is introduced."})
    return agent, lab, context


def run_case(source, output, case, *, calls=32):
    auth_mode = verify_codex_subscription()
    output = Path(output).resolve()
    agent, lab, context = restore_case(source, output, case, calls=calls)
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["auth_mode"] = auth_mode
    write_json(output / "protocol.json", protocol)
    agent.provider = CodexCodeProvider(agent.workspace, model="gpt-6-luna",
        reasoning_effort="medium", timeout_seconds=180)

    def checkpoint(data):
        latest = data["events"][-1] if len(data["events"]) > context["prefix_events"] else {}
        brief = {"study_status": "running", "case": case, "costs": data["costs"],
            "new_environment_steps": lab.environment_steps - context["costs_before"]["environment_steps"],
            "score": data["score"], "latest_event": {
                "index": len(data["events"]) - 1 if latest else None,
                "model_call": latest.get("model_call"), "action": latest.get("action"),
                "operator": latest.get("arguments", {}).get("operator"),
                "status": latest.get("feedback", {}).get("status"),
                "inquiry_question_id": latest.get("inquiry_question_id")}}
        write_json(output / "active.json", brief)
        print(json.dumps({"case": case, "calls": data["costs"]["model_calls"],
            "actions": lab.environment_steps, "completed": data["score"]["completedSuccessfully"]}), flush=True)

    agent._on_checkpoint = checkpoint
    checkpoint(account(agent))
    execution = agent._run_procedure(lab, METHOD_PATH, max_steps=256)
    agent._save_state()
    result = account(agent)
    changed_source = (agent.workspace / METHOD_PATH).read_text()
    (output / "source_after.py").write_text(changed_source)
    result.update(study_status="finished", episode_status=result["status"], execution=execution,
        costs_before=context["costs_before"], new_costs={key: value - context["costs_before"][key]
            for key, value in result["costs"].items()}, replay=context["replay"],
        declarations_before=context["declarations_before"], declarations_after=declarations(changed_source))
    write_json(output / "result.json", result)
    prefix = context["prefix_observations"]
    brief = {"study_status": "finished", "case": case, "status": execution["status"],
        "success": result["score"]["completedSuccessfully"], "prefix_actions": prefix,
        "total_actions": lab.environment_steps, "new_actions": lab.environment_steps - prefix,
        "costs": result["costs"], "new_costs": result["new_costs"],
        "declarations_before": result["declarations_before"],
        "declarations_after": result["declarations_after"],
        "inquiries": [{"id": key, "status": value.get("status"),
            "model_calls": value.get("model_calls")} for key, value in agent.llm_inquiries.items()],
        "new_evidence_answers": [question for question in agent.questions
            if any(index >= prefix for index in question.get("evidence", []))]}
    write_json(output / "brief.json", brief)
    write_json(output / "active.json", {"study_status": "finished", "case": case,
        "status": execution["status"], "score": result["score"], "costs": result["costs"]})
    return brief


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=(*CASES, "pair"), default="pair")
    parser.add_argument("--calls", type=int, default=32)
    args = parser.parse_args()
    selected = CASES if args.case == "pair" else (args.case,)
    for case in selected:
        output = args.output / case if args.case == "pair" else args.output
        brief = run_case(args.source, output, case, calls=args.calls)
        print(json.dumps({key: brief[key] for key in ("case", "status", "success", "new_actions")}), flush=True)
