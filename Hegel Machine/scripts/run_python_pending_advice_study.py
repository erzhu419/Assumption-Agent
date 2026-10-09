#!/usr/bin/env python3
"""Compare persistent research advice with event-only advice on new apparatuses."""

import argparse
from collections import Counter
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
from scripts.run_candidate_retention_cycle import task_for
from scripts.run_candidate_retention_cycle_llm import (
    execute_episode, prepare_episode, score_episode, train_source, write_json,
)
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription
from scripts.run_python_advice_ablation import (
    RecordingAdviceAgent, advice_evidence, protocol as advice_protocol,
)
from scripts.run_python_research_routing import operator_calls


class EventOnlyAdviceAgent(RecordingAdviceAgent):
    """Retain ordinary event advice while withholding the new persistence paths."""

    def _pending_python_research_advice(self, laboratory):
        return None

    def _state_research_advice(self):
        return None


def protocol():
    plan = advice_protocol()
    plan.update(study="python_pending_research_advice_new_apparatus_study",
        conditions=["persistent", "event_only"],
        cases=[{"apparatus_seed": 971106001 + 1009 * index,
                "changed_response_seed": 981106001 + 1009 * index,
                "seed": 991106001 + 1009 * index} for index in range(4)],
        source="Learn one source per new apparatus with live Luna and the common three-trajectory "
            "prefix. Only its actual freshly confirmed predictor is published. No supplied "
            "formula, source replacement, repeated source attempt, or terminal-based admission.",
        intervention="Persistent advice versus the preceding event-only advice. The experiment "
            "baseline disables only pending-advice fallback and the cached advice field in "
            "research_state. Both conditions retain original event feedback, explicit "
            "research_choice, all numerical operators, identical prompts and equal budgets.",
        accounting="Each of four sources is physically learned once. Report actual source "
            "plus all target costs, and each condition's target costs with shared source costs "
            "added once separately. Preserve incomplete usage rather than estimating tokens.",
        failure_policy="A source without fresh confirmation blocks its four dependent target "
            "episodes. Keep their registered slots, failed source costs and failure reason; "
            "do not replace sources or report only the selected successful sources. Target "
            "failures and service interruptions remain in the registered cohort.",
        adoption="Both conditions show original event advice. Record actual render opportunities "
            "privately; only the last recommendation in an event is matched against the next "
            "actual action. Pending advice after an unrelated action also counts as a new "
            "opportunity. Exact matching does not infer adoption from free text.",
        terminal_order="All source and target learning ends before any terminal evaluation. "
            "Source publication uses fresh confirmation only; terminal outcomes never filter jobs.",
        scope="Four new apparatus parameter realizations, each with one response change and "
            "one recurrent-instrument change; two target conditions per change. Four independent "
            "apparatus units, eight paired targets, sixteen registered target episodes, plus "
            "four shared sources. This tests advice persistence within existing families, "
            "not vanilla versus Hegel or an open-world capability claim.")
    plan.pop("previous_experiment", None)
    return plan


def target_jobs(sources, output, plan):
    """Keep all registered slots; source failure blocks both matched conditions."""
    jobs, skipped = [], []
    slot = 0
    for index, source in enumerate(sources):
        for kind_index, kind in enumerate(plan["target_kinds"]):
            order = plan["conditions"] if (index + kind_index) % 2 == 0 else plan["conditions"][::-1]
            for condition in order:
                entry = {"case": source["case"], "kind": kind, "condition": condition,
                         "path": Path(output) / f"trial_{slot:03d}", "source": source}
                if source["candidate"] is None:
                    skipped.append({key: str(value) if isinstance(value, Path) else value
                                    for key, value in entry.items() if key != "source"} | {
                        "reason": "source_not_freshly_confirmed",
                        "source_status": source["episode"]["training"]["status"]})
                else:
                    jobs.append(entry)
                slot += 1
    return jobs, skipped


def summarize(sources, results, skipped, plan):
    source_episodes = [source["episode"] for source in sources]
    source_costs = [episode["costs"] for episode in source_episodes]
    usage_keys = ("known_tokens", "input_tokens", "cached_input_tokens", "missing_usage_calls")
    conditions = {}
    for condition in plan["conditions"]:
        selected = [result for result in results if result["condition"] == condition]
        conditions[condition] = {
            "registered_targets": len(plan["cases"]) * len(plan["target_kinds"]),
            "attempted_targets": len(selected),
            "source_blocked_targets": sum(item["condition"] == condition for item in skipped),
            "target_statuses": dict(Counter(result["status"] for result in selected)),
            "confirmed_and_correct": sum(result["confirmed_and_correct"] for result in selected),
            "false_confirmations": sum(result["false_confirmation"] for result in selected),
            "target_costs": total_costs([result["costs"] for result in selected]),
            "source_plus_target_costs": total_costs(source_costs + [result["costs"] for result in selected]),
            **{key: sum(result[key] for result in selected) for key in usage_keys},
            "advice": {key: sum(result["advice"][key] for result in selected) for key in
                ("rendered_opportunities", "superseded_opportunities", "actionable_final_opportunities",
                 "opportunities_with_next_event", "visible_next_action_matches")}}
    all_episodes = source_episodes + results
    return {"study": plan["study"], "registered_sources": len(plan["cases"]),
        "registered_targets": len(plan["cases"]) * len(plan["target_kinds"]) * len(plan["conditions"]),
        "sources": sources, "results": results, "skipped": skipped, "conditions": conditions,
        "source_costs": total_costs(source_costs),
        "actual_source_plus_all_target_costs": total_costs([episode["costs"] for episode in all_episodes]),
        **{key: sum(episode[key] for episode in all_episodes) for key in usage_keys}, "scope": plan["scope"]}


def run_study(output, *, provider_factory, auth_mode=None, study_protocol=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = deepcopy(study_protocol) if study_protocol is not None else protocol()
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_python_advice_ablation.py", "run_python_research_routing.py",
                 "run_python_refit_diagnosis.py", "run_candidate_retention_cycle_llm.py",
                 "run_candidate_retention_cycle.py", "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)

    def source_job(index):
        return train_source(plan["cases"][index], output / f"source_{index:03d}", plan, provider_factory)

    def target_job(job):
        case, kind, path = job["case"], job["kind"], job["path"]
        settings = task_for(case, 1)
        if kind == "recurrent_instrument_change":
            settings.update(instrument_mode="recurrent", response_seed=case["apparatus_seed"])
        payload = prepare_episode(case, 1, path, [deepcopy(job["source"]["candidate"])], plan,
            provider=provider_factory(path), laboratory=TemporalInquiryLab(**settings),
            agent_class=RecordingAdviceAgent if job["condition"] == "persistent" else EventOnlyAdviceAgent)
        payload["agent"]._advice_learning = True
        return job, execute_episode(payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = list(pool.map(source_job, range(len(plan["cases"]))))
        jobs, skipped = target_jobs(sources, output, plan)
        write_json(output / "source_outcomes.json", {"skipped": skipped, "sources": [{
            "case": source["case"], "status": source["episode"]["training"]["status"],
            "candidate": source["candidate"], "costs": source["episode"]["training"]["costs"]}
            for source in sources]})
        trained = list(pool.map(target_job, jobs))
    # Publication has already happened; terminal labels cannot select sources.
    scored_sources = [{"case": source["case"], "candidate": source["candidate"],
                       "episode": score_episode(source["episode"], plan)} for source in sources]
    results = []
    for job, payload in trained:
        result = score_episode(payload, plan)
        events = payload["training"]["events"]
        result.update(case=job["case"], condition=job["condition"], kind=job["kind"],
            workspace=str(payload["path"]), operator_calls=operator_calls(events),
            advice=advice_evidence(payload["agent"], events, visible=True))
        write_json(output / "evaluations" / (payload["path"].name + ".json"), result)
        results.append(result)
    summary = summarize(scored_sources, results, skipped, plan)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_study(args.output, auth_mode=verify_codex_subscription(), provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180))
    print(json.dumps({key: value for key, value in summary.items() if key not in {"sources", "results"}}, indent=2))


if __name__ == "__main__":
    main()
