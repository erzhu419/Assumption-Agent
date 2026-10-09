#!/usr/bin/env python3
"""Matched live development replay with visible versus withheld research advice."""

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
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from scripts.run_candidate_retention_cycle import task_for
from scripts.run_candidate_retention_cycle_llm import (
    action_matches, execute_episode, prepare_episode, score_episode, write_json,
)
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription
from scripts.run_python_refit_diagnosis import SOURCE
from scripts.run_python_research_routing import operator_calls, protocol as routing_protocol


class RecordingAdviceAgent(TemporalOperatorAgent):
    """Keep an experiment-private record outside the model's evidence files."""

    def _execute(self, action, arguments, laboratory):
        previous = getattr(self, "_advice_action", None)
        self._advice_action = {"action": action, "operator": arguments.get("operator")}
        try:
            return super()._execute(action, arguments, laboratory)
        finally:
            self._advice_action = previous

    def _render_research_advice(self, advice):
        if not hasattr(self, "_advice_ledger"):
            self._advice_ledger = []
        self._advice_ledger.append({"event_index": len(self.events),
            "learning": getattr(self, "_advice_learning", False),
            "trigger": deepcopy(getattr(self, "_advice_action", None)), "advice": deepcopy(advice)})
        return super()._render_research_advice(advice)


class NoAdviceAgent(RecordingAdviceAgent):
    def _render_research_advice(self, advice):
        super()._render_research_advice(advice)
        return None


def protocol():
    plan = routing_protocol("live")
    plan.update(study="python_research_advice_ablation_development_replay",
        conditions=["advice", "no_advice"],
        intervention="Both conditions compute research advice from already paid results. "
            "The experiment-only no_advice subclass returns None from the rendering hook, "
            "before feedback, operator results or ordinary files are saved. It retains the "
            "same explicit research_choice, refit and other tools. Advice is shown after "
            "evaluation, failed confirmation and numerical operator results, including "
            "successful repair steps that enable later evaluation or submission.",
        controller="Every action after the matched three-trajectory prefix is selected by "
            "Luna. Neither condition forces an initial action or follows advice automatically. "
            "Same model, prompt, tools, budgets and confirmation seeds; maximum 12 model calls.",
        adoption="Record the actual recommended next action at the rendering hook privately. "
            "For multiple recommendations within one event, earlier ones are superseded; "
            "only the final advice is compared with the next actual event, using exact "
            "action/required-argument matching. A match in no_advice is counterfactual "
            "alignment, not adoption of an unseen recommendation. No inference from text.",
        accounting="Report actual token/cache usage, observations, computation and execution "
            "for each condition. Shared existing source learning is reported separately "
            "and is not consumed again. Advice computes no new numerical fit or observation.",
        scope="Eight live episodes on four previously examined development tasks. The "
            "paired intervention tests visibility of concrete advice, not a new independent "
            "task cohort or a complete vanilla-versus-Hegel comparison.")
    plan.pop("offline_choice_round_limit", None)
    return plan


def advice_evidence(agent, events, *, visible):
    ledger = deepcopy(getattr(agent, "_advice_ledger", []))
    latest = {}
    for index, item in enumerate(ledger):
        if item["learning"]:
            latest[item["event_index"]] = index
    for index, item in enumerate(ledger):
        item["superseded"] = item["learning"] and latest[item["event_index"]] != index
        step = item["event_index"]
        action = (item["advice"] or {}).get("next_action")
        eligible = item["learning"] and not item["superseded"] and bool(action)
        item["actionable_final_advice"] = eligible
        item["next_event_exists"] = eligible and step + 1 < len(events)
        item["next_action_matches"] = bool(item["next_event_exists"] and
                                             action_matches(events[step + 1], action))
        item["advice_visible"] = visible
    return {"advice_ledger": ledger,
        "rendered_opportunities": sum(item["learning"] for item in ledger),
        "superseded_opportunities": sum(item["superseded"] for item in ledger),
        "actionable_final_opportunities": sum(item["actionable_final_advice"] for item in ledger),
        "opportunities_with_next_event": sum(item["next_event_exists"] for item in ledger),
        "visible_next_action_matches": sum(item["next_action_matches"] for item in ledger) if visible else 0,
        "counterfactual_next_action_matches": sum(item["next_action_matches"] for item in ledger) if not visible else 0}


def run_study(output, *, provider_factory, auth_mode=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = protocol()
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_python_research_routing.py", "run_python_refit_diagnosis.py",
                 "run_candidate_retention_cycle_llm.py", "run_candidate_retention_cycle.py",
                 "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    sources, jobs = [], []
    for case_index, case in enumerate(plan["cases"]):
        path = SOURCE / f"case_{case['apparatus_seed']}" / "source/episode_000"
        candidate = json.loads((path / "published_candidate.json").read_text())["candidate"]
        previous = json.loads((path / "evaluation.json").read_text())
        sources.append({"path": str(path), "candidate": candidate, "costs": previous["costs"]})
        for kind_index, kind in enumerate(plan["target_kinds"]):
            order = plan["conditions"] if (case_index + kind_index) % 2 == 0 else plan["conditions"][::-1]
            for condition in order:
                jobs.append((case, kind, condition, candidate, output / f"trial_{len(jobs):03d}"))

    def train(job):
        case, kind, condition, candidate, path = job
        settings = task_for(case, 1)
        if kind == "recurrent_instrument_change":
            settings.update(instrument_mode="recurrent", response_seed=case["apparatus_seed"])
        payload = prepare_episode(case, 1, path, [deepcopy(candidate)], plan,
            provider=provider_factory(path), laboratory=TemporalInquiryLab(**settings),
            agent_class=RecordingAdviceAgent if condition == "advice" else NoAdviceAgent)
        payload["agent"]._advice_learning = True
        return condition, kind, execute_episode(payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        episodes = list(pool.map(train, jobs))
    # Keep the private advice ledger outside all learning workspaces until every
    # episode stops. Terminal labels are also produced only after this point.
    results = []
    for condition, kind, payload in episodes:
        result = score_episode(payload, plan)
        events = payload["training"]["events"]
        result.update(condition=condition, kind=kind, workspace=str(payload["path"]),
            operator_calls=operator_calls(events),
            advice=advice_evidence(payload["agent"], events, visible=condition == "advice"))
        results.append(result)
        write_json(output / "evaluations" / (payload["path"].name + ".json"), result)
    conditions = {}
    for condition in plan["conditions"]:
        selected = [result for result in results if result["condition"] == condition]
        conditions[condition] = {"attempted_targets": len(selected),
            "confirmed_and_correct": sum(r["confirmed_and_correct"] for r in selected),
            "false_confirmations": sum(r["false_confirmation"] for r in selected),
            "new_target_costs": total_costs([r["costs"] for r in selected]),
            **{key: sum(r[key] for r in selected) for key in
               ("known_tokens", "input_tokens", "cached_input_tokens", "missing_usage_calls")},
            "advice": {key: sum(r["advice"][key] for r in selected) for key in
                ("rendered_opportunities", "superseded_opportunities", "actionable_final_opportunities",
                 "opportunities_with_next_event", "visible_next_action_matches", "counterfactual_next_action_matches")}}
    summary = {"study": plan["study"], "registered_episodes": len(jobs), "sources": sources,
        "existing_source_costs": total_costs([source["costs"] for source in sources]),
        "results": results, "conditions": conditions,
        "actual_new_target_costs": total_costs([r["costs"] for r in results]), "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = verify_codex_subscription()
    summary = run_study(args.output, auth_mode=auth, provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180))
    print(json.dumps({key: value for key, value in summary.items() if key not in {"sources", "results"}}, indent=2))


if __name__ == "__main__":
    main()
