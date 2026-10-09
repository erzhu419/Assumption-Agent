#!/usr/bin/env python3
"""Execute the real source-feedback action split under one fixed budget.

The preparation is the existing ``source_competition`` fixture.  Both arms
receive the same two already-screened offers and the same paid predecessor.
Only the measured ``research_return`` records differ.  The candidate arm
therefore withdraws the bad successor and executes the repaired one; the
matched arm has no return evidence and executes the old bad successor.  The
visit is performed through ``UnifiedResearchSession._perform_revisit`` so the
receipt includes the actual continuation work and its paid costs.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUT = ROOT / "artifacts" / "source_feedback_same_budget_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TESTS))


def _fixture():
    from test_cognition_unified_source_returns import failed_successor, source_competition

    case = failed_successor.__wrapped__()
    competition = source_competition.__wrapped__(case)
    return case, *competition[1:]


def _config():
    from hegel_machine.cognition.models import ResearchConfig

    return ResearchConfig(
        observation_budget=100,
        compute_budget=20_000,
        minimum_improvement=1e-7,
        max_candidates=16,
        max_rounds=1,
        active_probes=False,
        risk_learning=False,
        seed=1237,
    )


def _offer(case, source_id, library):
    from hegel_machine.cognition.models import SearchBudget
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_revisits import offer_successor_knowledge

    previous = deepcopy(case["previous"])
    previous["task_id"] = "same_budget_target"
    # The real fixture lab has no channels and returns deterministic rows.
    lab = type(case["lab"])(case["spec"])
    offer = offer_successor_knowledge(
        UnifiedResearchAgent(_config(), library=library), lab, previous,
        {source_id}, SearchBudget(4_000))
    if offer is None:
        raise RuntimeError(f"real continuation offer missing for {source_id}")
    offer.update(task_position=0, source_task_position=2)
    return previous, offer


def _prepare():
    from hegel_machine.cognition.memory import KnowledgeLibrary

    case, good, warm, _cold = _fixture()
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(warm.library.to_dict())))
    previous, bad_offer = _offer(case, case["source_id"], library)
    _, good_offer = _offer(case, good, library)
    return {
        "case_id": "source_competition_real_same_budget",
        "source_ids": {"bad": case["source_id"], "good": good},
        "previous": previous,
        "offers": {"bad": bad_offer, "good": good_offer},
        "library": library.to_dict(),
        "spec": asdict(case["spec"]),
        "config": asdict(_config()),
        "measurement_budget": 0,
        "external_model_calls": 0,
    }


def _restore_spec(payload):
    from hegel_machine.cognition.models import Probe, TaskSpec

    data = dict(payload)
    data["feature_names"] = tuple(data["feature_names"])
    data["initial_probes"] = tuple(Probe(item["probe_id"], tuple(item["sequence"]))
                                   for item in data["initial_probes"])
    data["available_probes"] = tuple(Probe(item["probe_id"], tuple(item["sequence"]))
                                     for item in data["available_probes"])
    data["context"] = dict(data.get("context", {}))
    data["feature_bounds"] = {key: tuple(value) for key, value in data.get("feature_bounds", {}).items()}
    return TaskSpec(**data)


def _losses(report, case):
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from test_cognition_conflict_transfer_refit import conflict_learning

    _source_spec, _target, _source, _old, _fresh, later = conflict_learning()
    program = Program.from_dict(report["program"])
    return {
        "paid_revisit_loss": loss(program, case["lab"].observed, report["fallback"], .5),
        "held_out_later_loss": loss(program, later, report["fallback"], .5),
    }


def _run_arm(prepared, arm):
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.unified_session import UnifiedResearchSession

    library_payload = deepcopy(prepared["library"])
    if arm == "matched_no_return":
        for concept in library_payload["concepts"]:
            concept["reuse_history"] = [
                row for row in concept.get("reuse_history", [])
                if row.get("kind") != "research_return"
            ]
    library = KnowledgeLibrary.from_dict(library_payload)
    config = __import__("hegel_machine.cognition.models", fromlist=["ResearchConfig"]).ResearchConfig(
        **prepared["config"])
    spec = _restore_spec(prepared["spec"])
    from test_cognition_unified_source_returns import failed_successor

    # Recreate the real deterministic fixture lab and add only the context
    # manager required by UnifiedSession.  No rows, gains or returns are
    # injected into the continuation.
    case, _good, _warm, _cold = _fixture()
    base = type(case["lab"])

    class SessionLab(base):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    session = UnifiedResearchSession(
        config,
        tasks=[{"task_index": 0, "seed": config.seed}],
        library=library,
        measurement_budget=prepared["measurement_budget"],
        lab_factory=lambda _item: SessionLab(spec),
    )
    session.state["results"] = [{"task_position": 0, "research": deepcopy(prepared["previous"])}]
    session.state["pending_revisits"] = [deepcopy(prepared["offers"]["good" if arm == "candidate" else "bad"])]
    before = session._costs()
    offer = session.state["pending_revisits"][0]
    ok = session._perform_revisit(offer, allocation_limits=offer["minimum_resources"])
    if not ok:
        raise RuntimeError(f"{arm} continuation was not executed")
    report = session.state["results"][0]["research"]
    after = session._costs()
    return {
        "arm": arm,
        "executed_source": offer["candidate"]["source_concept_ids"],
        "status": report["status"],
        "losses": _losses(report, case),
        "visit_costs": report["visit_costs"],
        "session_cost_delta": {key: after[key] - before[key] for key in before},
        "revisits": len(session.state["revisits"]),
        "scheduler_events": session.state["scheduler_events"],
        "external_model_calls": 0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prepared_path = OUT / "prepared.json"
    results_path = OUT / "results.json"
    prepared = _prepare()
    prepared_path.write_text(json.dumps(prepared, indent=2, ensure_ascii=False) + "\n")
    results = {
        "status": "complete",
        "protocol": "same_screened_offers_same_paid_prefix_same_execution_limits",
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
