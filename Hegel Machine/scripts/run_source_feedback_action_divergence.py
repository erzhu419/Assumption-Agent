#!/usr/bin/env python3
"""Record the real mixed-source action divergence without running a long study.

The fixture supplies a measured negative transfer and a repaired source.  The
candidate arm keeps both real returns; the matched arm removes only those
return records while keeping the concepts and the already screened offers.
Both arms then run the same zero-observation scheduler step.
"""

from copy import deepcopy
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUT = ROOT / "artifacts" / "source_feedback_action_divergence_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TESTS))


def _actual_fixture():
    from test_cognition_unified_source_returns import failed_successor, source_competition

    case = failed_successor.__wrapped__()
    competition = source_competition.__wrapped__(case)
    return (case, *competition[1:])


def _config():
    from hegel_machine.cognition.models import ResearchConfig

    return ResearchConfig(
        observation_budget=100,
        compute_budget=20_000,
        minimum_improvement=1e-7,
        max_candidates=16,
        active_probes=False,
        risk_learning=False,
        seed=1237,
    )


def _prepare():
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import SearchBudget
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_revisits import offer_successor_knowledge

    case, good, warm, _cold = _actual_fixture()
    config = _config()
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(warm.library.to_dict())))

    positive_previous = deepcopy(case["previous"])
    positive_previous["task_id"] = "positive_source_target"
    positive_lab = type(case["lab"])(case["spec"])
    positive_agent = UnifiedResearchAgent(config, library=library, measurement_budget=0)
    positive_offer = offer_successor_knowledge(
        positive_agent, positive_lab, positive_previous, {good}, SearchBudget(4_000))
    if positive_offer is None:
        raise RuntimeError("real repaired-source continuation was not offered")
    positive_report = positive_agent.run(
        positive_lab, previous=positive_previous,
        continuation_candidate=positive_offer["candidate"])
    positive_return = next(
        row for row in positive_report["knowledge_return"]["returns"]
        if row["source_id"] == good and row["retained"] and row["paired_gain"] > 0)

    pending_previous = deepcopy(case["previous"])
    pending_previous["task_id"] = "pending_source_target"
    pending = []
    for position, source_id in enumerate((case["source_id"], good)):
        lab = type(case["lab"])(case["spec"])
        agent = UnifiedResearchAgent(config, library=library, measurement_budget=0)
        offer = offer_successor_knowledge(
            agent, lab, pending_previous, {source_id}, SearchBudget(4_000))
        if offer is None:
            raise RuntimeError(f"real pending offer missing for {source_id}")
        offer.update(task_position=position, source_task_position=2)
        pending.append(offer)

    evaluation_lab = type(case["lab"])(case["spec"])
    evaluation_rows = []
    for probe in case["spec"].available_probes:
        evaluation_rows.extend(evaluation_lab.observe(probe))

    return {
        "case_id": "source_competition_real_mixed",
        "source_ids": {"bad": case["source_id"], "good": good},
        "positive_return": {
            "source_id": positive_return["source_id"],
            "condition": positive_return["condition"],
            "paired_gain": positive_return["paired_gain"],
            "retained_normalized_gain": positive_return["retained_normalized_gain"],
            "adaptation_compute": positive_return["adaptation_compute"],
            "comparison_compute": positive_return["comparison_compute"],
        },
        "pending": pending,
        "previous": deepcopy(pending_previous),
        "evaluation_rows": [asdict(row) for row in evaluation_rows],
        "library": library.to_dict(),
        "config": {**asdict(config), "measurement_budget": 0},
        "resource_pool": {
            "observation_cost": 100,
            "measurement_cost": 100,
            "compute_cost": 10_000,
        },
        "external_model_calls": 0,
    }


def _summary_offer(offer):
    candidate = offer["candidate"]
    return {
        "source_concept_ids": list(candidate.get("source_concept_ids", [])),
        "task_position": offer["task_position"],
        "expected_loss_reduction": offer["expected_loss_reduction"],
        "relative_gain": offer.get("relative_gain"),
        "minimum_resources": dict(offer["minimum_resources"]),
    }


def _run_arm(prepared, arm):
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
    from hegel_machine.cognition.research_portfolio import choose_research_visit
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_session import UnifiedResearchSession

    library_payload = deepcopy(prepared["library"])
    if arm == "matched_no_return":
        for concept in library_payload["concepts"]:
            concept["reuse_history"] = [
                row for row in concept.get("reuse_history", [])
                if row.get("kind") != "research_return"]
    library = KnowledgeLibrary.from_dict(library_payload)
    config = ResearchConfig(**{
        key: value for key, value in prepared["config"].items()
        if key != "measurement_budget"})
    session = UnifiedResearchSession(
        config,
        tasks=[{"task_index": 0, "seed": 1237}, {"task_index": 0, "seed": 1238}],
        library=library,
        measurement_budget=prepared["config"]["measurement_budget"],
    )
    session.state["pending_revisits"] = deepcopy(prepared["pending"])
    before_screening = session.state["screening_compute_cost"]
    session.lab_factory = lambda item: (_ for _ in ()).throw(
        AssertionError("source feedback must not reopen a laboratory"))
    session._refresh_source_feedback()

    def allocate(offer, others, remaining):
        return dict(offer["minimum_resources"])

    remaining = dict(prepared["resource_pool"])
    budget = SearchBudget(remaining["compute_cost"])
    choice = choose_research_visit(
        session.state["pending_revisits"], remaining, budget, allocation_for=allocate)
    if choice is None or choice.get("offer") is None:
        raise RuntimeError(f"{arm} produced no executable selected visit")

    # Execute the selected first visit with the same fixed budget and a fresh
    # laboratory. Evaluation rows come from held-out probes and are never
    # used to construct the pending offers.
    case, _good, _warm, _cold = _actual_fixture()
    agent = UnifiedResearchAgent(config, library=library, measurement_budget=0)
    laboratory = type(case["lab"])(case["spec"])
    report = agent.run(
        laboratory,
        previous=deepcopy(prepared["previous"]),
        continuation_candidate=deepcopy(choice["offer"]["candidate"]),
    )
    rows = [
        Observation(row["features"], row["target"], tuple(row["history"]))
        for row in prepared["evaluation_rows"]
    ]
    program = Program.from_dict(report["program"])
    heldout_loss = loss(program, rows, report["fallback"], .5)
    return {
        "arm": arm,
        "pending_before": [_summary_offer(offer) for offer in prepared["pending"]],
        "feedback_events": [
            {key: event[key] for key in (
                "action", "source_concept_ids", "matching_returns", "compute_cost",
                "knowledge_return_score") if key in event}
            for event in session.state["scheduler_events"]],
        "pending_after": [{
            **_summary_offer(offer),
            "source_return_status": offer.get("source_return_status"),
            "knowledge_return_matches": offer.get("knowledge_return_matches", 0),
            "knowledge_return_score": offer.get("knowledge_return_score", 0.),
        } for offer in session.state["pending_revisits"]],
        "selected_source": (
            choice["offer"]["candidate"]["source_concept_ids"]
            if choice and choice.get("offer") else None),
        "selected_indices": choice["decision"]["selected_indices"] if choice else None,
        "selected_knowledge_return_score": (
            choice["decision"].get("selected_knowledge_return_score") if choice else None),
        "screening_compute_delta": session.state["screening_compute_cost"] - before_screening,
        "scheduler_search_compute": budget.spent,
        "resource_pool": remaining,
        "lab_reopens": 0,
        "executed_visit": {
            "source_concept_ids": list(choice["offer"]["candidate"].get("source_concept_ids", [])),
            "status": report["status"],
            "visit_costs": report["visit_costs"],
            "heldout_rows": len(rows),
            "heldout_loss": heldout_loss,
            "program": report["program"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "run", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prepared_path = OUT / "prepared.json"
    results_path = OUT / "results.json"
    if args.phase in ("prepare", "all"):
        prepared = _prepare()
        prepared_path.write_text(json.dumps(prepared, indent=2, ensure_ascii=False) + "\n")
    else:
        prepared = json.loads(prepared_path.read_text())
    if args.phase in ("run", "all"):
        results = {
            "status": "complete",
            "prepared": str(prepared_path.relative_to(ROOT)),
            "arms": {
                "candidate": _run_arm(prepared, "candidate"),
                "matched_no_return": _run_arm(prepared, "matched_no_return"),
            },
            "external_model_calls": 0,
        }
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
