#!/usr/bin/env python3
"""Compare a current-feedback procedure option with the same interface-only tool."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent
from scripts.run_candidate_retention_cycle_llm import action_matches, execute_episode, score_episode, write_json
from scripts.run_online_inquiry_llm import verify_codex_subscription
from scripts.run_python_procedure_study import (
    PROGRAM_PATH, SOURCE_OUTCOMES, execution_evidence, prepare_target, protocol as procedure_protocol,
    summarize as procedure_summary, target_jobs,
)


PROCEDURE_OFFER_INTERFACE = """When research remains open, feedback may offer this complete action as
research_advice.procedure_action or available_procedure_action. It continues
the current investigation through revision and fresh confirmation; next_action
remains the individual next step. You choose which action to execute.
"""


class FeedbackOfferAgent(ProcedureConsumerAgent):
    """Retain the tested offer as an experiment after its negative live result."""

    common_interface = ProcedureConsumerAgent.common_interface + PROCEDURE_OFFER_INTERFACE

    def _render_procedure_offer(self, offer):
        return offer

    def _procedure_offer(self):
        if self.model is not None or not (self.workspace / PROGRAM_PATH).is_file():
            return None
        return self._render_procedure_offer({"action": "run", "arguments": {
            "operator": "procedure", "path": PROGRAM_PATH, "max_steps": 12, "confirm": True}})

    def _render_research_advice(self, advice):
        rendered = super()._render_research_advice(advice)
        next_action = (rendered or {}).get("next_action")
        if next_action and next_action["action"] != "finish":
            offer = self._procedure_offer()
            if offer is not None:
                rendered = {**rendered, "procedure_action": offer}
        return rendered

    def _execute(self, action, arguments, laboratory):
        result = super()._execute(action, arguments, laboratory)
        if action == "run" and arguments.get("operator") == "procedure":
            return result
        next_action = result.get("next_action") or result.get("available_research_action")
        if not result.get("research_advice") and next_action and next_action["action"] != "finish":
            offer = self._procedure_offer()
            if offer is not None:
                result["available_procedure_action"] = offer
        return result


class InterfaceOnlyAgent(FeedbackOfferAgent):
    """Keep the complete program/tool, withholding only its current-feedback offer."""

    def _render_procedure_offer(self, offer):
        return None


def protocol(sources):
    plan = procedure_protocol("live", sources)
    plan.update(study="python_procedure_feedback_offer_development_replay",
        conditions=["feedback_offer", "interface_only"],
        intervention="Both conditions have the identical complete procedure tool, editable "
            "shared program, common interface and atomic next_action recommendations. Only "
            "feedback_offer renders the optional procedure_action in research_advice or "
            "available_procedure_action in feedback. Interface-only suppresses this option "
            "through the rendering hook before it enters saved feedback or research state. "
            "Neither condition automatically executes a recommendation.",
        adoption="Count actual outer procedure requests and executed method steps separately. "
            "Match each visible feedback option to the immediately following action using "
            "exact action and required-argument matching. A completed procedure must report "
            "completed plus passed fresh confirmation; target correctness still requires "
            "the common terminal evaluation. Interface-only may independently choose the tool.",
        scope="Six previously examined development targets, paired across feedback_offer and "
            "interface_only. Both use the same three actual source archives and paid prefixes. "
            "This tests presenting an available procedure as a current option, not new-task "
            "generalization or discovery of a learned research method.")
    return plan


def prepare_offer_target(job, plan, provider=None):
    cls = FeedbackOfferAgent if job["condition"] == "feedback_offer" else InterfaceOnlyAgent
    # The shared helper's bundled preparation supplies exactly the same complete
    # tool/interface to both conditions. Their study labels stay outside the agent.
    payload = prepare_target({**job, "condition": "bundled"}, plan, provider, agent_class=cls)
    payload["agent"].common_interface += PROCEDURE_OFFER_INTERFACE
    return payload


def offer_evidence(events):
    offers, invocations = [], []
    for index, event in enumerate(events):
        feedback = event.get("feedback", {})
        option = (feedback.get("research_advice") or {}).get("procedure_action")
        if option is None:
            option = feedback.get("available_procedure_action")
        if option is not None:
            has_next = index + 1 < len(events)
            offers.append({"event_index": index, "action": deepcopy(option), "has_next_event": has_next,
                           "next_action_matches": has_next and action_matches(events[index + 1], option)})
        if event.get("action") == "run" and event.get("arguments", {}).get("operator") == "procedure":
            invocations.append({"event_index": index, "arguments": deepcopy(event["arguments"]),
                "status": feedback.get("status"), "steps": len(feedback.get("steps", [])),
                "fresh_confirmed": bool((feedback.get("fresh_confirmation") or {}).get("passed")),
                "completed_with_fresh_confirmation": feedback.get("status") == "completed" and
                    bool((feedback.get("fresh_confirmation") or {}).get("passed"))})
    return {"visible_offer_events": len(offers),
        "offers_with_next_event": sum(item["has_next_event"] for item in offers),
        "next_action_matches": sum(item["next_action_matches"] for item in offers),
        "procedure_invocations": len(invocations),
        "procedure_internal_steps": sum(item["steps"] for item in invocations),
        "procedures_completed_with_fresh_confirmation": sum(
            item["completed_with_fresh_confirmation"] for item in invocations),
        "offers": offers, "invocations": invocations}


def summarize(sources, results, plan):
    summary = procedure_summary(sources, results, plan)

    def adoption(items):
        keys = ("visible_offer_events", "offers_with_next_event", "next_action_matches",
                "procedure_invocations", "procedure_internal_steps", "procedures_completed_with_fresh_confirmation")
        return {key: sum(item["procedure_offer"][key] for item in items) for key in keys}

    for condition, aggregate in summary["conditions"].items():
        selected = [item for item in results if item["condition"] == condition]
        aggregate["procedure_offer"] = adoption(selected)
        for kind, subtotal in aggregate["by_kind"].items():
            subtotal["procedure_offer"] = adoption([item for item in selected if item["kind"] == kind])
    summary["actual_new_targets"]["procedure_offer"] = adoption(results)
    return summary


def run_study(output, *, provider_factory, auth_mode=None):
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    plan = protocol(sources)
    plan["provider"]["auth_mode"] = auth_mode
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_python_procedure_study.py", "run_python_pending_advice_study.py",
                 "run_python_advice_ablation.py", "run_python_research_routing.py", "run_python_refit_diagnosis.py",
                 "run_candidate_retention_cycle_llm.py", "run_candidate_retention_cycle.py",
                 "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        destination = output / "source_at_run/scripts" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, destination)
    jobs = target_jobs(sources, output, plan)

    def train(job):
        payload = prepare_offer_target(job, plan, provider_factory(job["path"]))
        return job, execute_episode(payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, jobs))
    results = []
    # All twelve learning episodes stop before terminal evaluation begins.
    for job, payload in trained:
        result = score_episode(payload, plan)
        result.update(case=job["case"], kind=job["kind"], condition=job["condition"], workspace=str(payload["path"]),
            **execution_evidence(payload["agent"], payload["training"]),
            procedure_offer=offer_evidence(payload["training"]["events"]))
        write_json(output / "evaluations" / (payload["path"].name + ".json"), result)
        results.append(result)
    summary = summarize(sources, results, plan)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_study(args.output, auth_mode=verify_codex_subscription(), provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180))
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"existing_sources", "results"}}, indent=2))


if __name__ == "__main__":
    main()
