#!/usr/bin/env python3
"""Small offline retain-vs-no-return study over real source-return fixtures.

Each trial rebuilds the existing ``failed_successor``/``source_competition``
fixture, creates the same screened bad/good offers, and gives both arms the
same continuation budget.  The feedback arm applies the measured source
return before choosing; the matched arm removes return records and follows
the old queue-first choice.  Conditions differ only in the target's observed
scale bounds, and task ids/seeds are repeated to expose stability.

This is a mechanism study, not a claim of scheduler optimality: it keeps the
legacy queue-first comparator explicit because the current offers have a
material predicted-gain ordering.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_multitask_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


SEEDS = (1237, 1238, 1239)
CONDITIONS = {
    "matching_bounds": {"x": (-2.25, 2.25), "z": (-2.25, 2.25)},
    "narrow_bounds": {"x": (-1.5, 1.5), "z": (-1.5, 1.5)},
}
TASK_IDS = ("feedback_task_a", "feedback_task_b")


def _fixture():
    from test_cognition_unified_source_returns import failed_successor, source_competition

    case = failed_successor.__wrapped__()
    good, warm, _cold = source_competition.__wrapped__(case)[1:]
    return case, good, warm.library


def _config(seed):
    from hegel_machine.cognition.models import ResearchConfig

    return ResearchConfig(observation_budget=100, compute_budget=20_000,
        minimum_improvement=1e-7, max_candidates=16, max_rounds=1,
        active_probes=False, risk_learning=False, seed=seed)


def _spec(case, condition, task_id):
    return replace(case["spec"], task_id=task_id,
        feature_bounds={key: tuple(value) for key, value in CONDITIONS[condition].items()})


def _offers(case, good, library, spec, config):
    from hegel_machine.cognition.models import SearchBudget
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_revisits import offer_successor_knowledge

    previous = deepcopy(case["previous"])
    previous["task_id"] = spec.task_id
    offers = []
    for source_id in (case["source_id"], good):
        lab = type(case["lab"])(spec)
        offer = offer_successor_knowledge(UnifiedResearchAgent(config, library=library),
            lab, previous, {source_id}, SearchBudget(4_000))
        if offer is None:
            raise RuntimeError(f"real continuation offer missing for {source_id} in {spec.task_id}")
        offers.append(offer)
    for position, offer in enumerate(offers):
        offer.update(task_position=position, source_task_position=2)
    return previous, offers


def _run_trial(case, good, base_library, *, condition, task_id, seed, arm):
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    from test_cognition_conflict_transfer_refit import conflict_learning

    config = _config(seed)
    spec = _spec(case, condition, task_id)
    source_library = KnowledgeLibrary.from_dict(json.loads(json.dumps(base_library.to_dict())))
    previous, offers = _offers(case, good, source_library, spec, config)
    library_payload = source_library.to_dict()
    if arm == "matched_no_return":
        for concept in library_payload["concepts"]:
            concept["reuse_history"] = [row for row in concept.get("reuse_history", [])
                if row.get("kind") != "research_return"]
    library = KnowledgeLibrary.from_dict(library_payload)
    session = UnifiedResearchSession(config, tasks=[{"task_index": 0, "seed": seed}],
        library=library, measurement_budget=0, lab_factory=lambda _item: type(case["lab"])(spec))
    session.state["pending_revisits"] = deepcopy(offers)
    session.state["results"] = [{"task_position": 0, "research": deepcopy(previous)}]
    session._refresh_source_feedback()
    pending = list(session.state["pending_revisits"])
    if not pending:
        raise RuntimeError(f"all offers withdrawn in {condition}/{task_id}/{arm}")
    # The candidate policy uses the first surviving offer.  The matched arm is
    # the historical queue-first policy, so its first item is the bad source.
    chosen = pending[0]
    laboratory = type(case["lab"])(spec)
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(
        laboratory, previous=deepcopy(previous),
        continuation_candidate=deepcopy(chosen["candidate"]))
    _source_spec, _target, _source, _old, _fresh, heldout = conflict_learning()
    program = Program.from_dict(report["program"])
    heldout_loss = loss(program, heldout, report["fallback"], .5)
    return {
        "condition": condition, "task_id": task_id, "seed": seed, "arm": arm,
        "selected_source": list(chosen["candidate"]["source_concept_ids"]),
        "pending_before": [list(item["candidate"]["source_concept_ids"]) for item in offers],
        "pending_after": [list(item["candidate"]["source_concept_ids"]) for item in pending],
        "feedback_events": [{key: event[key] for key in ("action", "matching_returns",
            "source_concept_ids", "compute_cost") if key in event}
            for event in session.state["scheduler_events"]],
        "heldout_later_loss": heldout_loss,
        "status": report["status"],
        "visit_costs": dict(report["visit_costs"]),
        "external_model_calls": 0,
    }


def main():
    rows = []
    for condition in CONDITIONS:
        for task_id in TASK_IDS:
            for seed in SEEDS:
                case, good, library = _fixture()
                for arm in ("feedback", "matched_no_return"):
                    rows.append(_run_trial(case, good, library, condition=condition,
                        task_id=task_id, seed=seed, arm=arm))
    for condition in CONDITIONS:
        for task_id in TASK_IDS:
            for seed in SEEDS:
                pair = [row for row in rows if row["condition"] == condition
                    and row["task_id"] == task_id and row["seed"] == seed]
                assert len(pair) == 2
                assert pair[0]["visit_costs"] == pair[1]["visit_costs"]
    matching = [row for row in rows if row["condition"] == "matching_bounds"
        and row["arm"] == "feedback"]
    narrow = [row for row in rows if row["condition"] == "narrow_bounds"
        and row["arm"] == "feedback"]
    assert all(row["selected_source"] == ["concept_0002"] for row in matching)
    assert all(row["selected_source"] == ["concept_0001"] for row in narrow)
    by_arm = {}
    for arm in ("feedback", "matched_no_return"):
        subset = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {"trials": len(subset),
            "selected_source_counts": {
                source: sum(source in row["selected_source"] for row in subset)
                for source in ("concept_0001", "concept_0002")},
            "heldout_later_loss_mean": statistics.mean(row["heldout_later_loss"] for row in subset),
            "heldout_later_loss_max": max(row["heldout_later_loss"] for row in subset),
            "resolved_count": sum(row["status"] == "completed" for row in subset),
            "feedback_event_count": sum(len(row["feedback_events"]) for row in subset),
            "external_model_calls": 0}
    by_condition = {}
    for condition in CONDITIONS:
        by_condition[condition] = {}
        for arm in ("feedback", "matched_no_return"):
            subset = [row for row in rows if row["condition"] == condition and row["arm"] == arm]
            by_condition[condition][arm] = {
                "trials": len(subset),
                "selected_source_counts": {
                    source: sum(source in row["selected_source"] for row in subset)
                    for source in ("concept_0001", "concept_0002")},
                "heldout_later_loss_mean": statistics.mean(
                    row["heldout_later_loss"] for row in subset),
                "resolved_count": sum(row["status"] == "completed" for row in subset),
                "feedback_event_count": sum(len(row["feedback_events"]) for row in subset),
            }
    payload = {"status": "complete",
        "protocol": {"conditions": list(CONDITIONS), "task_ids": list(TASK_IDS),
            "seeds": list(SEEDS), "same_screened_offers": True,
            "same_continuation_budgets": True, "offline_fixture_only": True,
            "matched_arm": "return_records_removed_then_queue_first",
            "feedback_arm": "negative_source_return_refresh_then_queue_first",
        "external_model_calls": 0}, "summary": by_arm,
        "summary_by_condition": by_condition, "trials": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
