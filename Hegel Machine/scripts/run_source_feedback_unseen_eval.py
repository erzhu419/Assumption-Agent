#!/usr/bin/env python3
"""Evaluate real good/bad successor continuations on a held-out trajectory.

Both arms use the same paid prefix, laboratory, and fixed research budgets.
The source concepts and executable refit recipes come from the existing
``failed_successor``/``source_competition`` fixtures.  The later trajectory is
the fixture's untouched response stream and is used only after the research
run, so it cannot affect the continuation choice.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_unseen_eval_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def _fixture():
    from test_cognition_conflict_transfer_refit import conflict_learning
    from test_cognition_unified_source_returns import failed_successor, source_competition

    case = failed_successor.__wrapped__()
    # This constructs the repaired concept from the actual paid rows and
    # retains the original transferred concept.  No score is injected.
    good, _warm, cold = source_competition.__wrapped__(case)[1:]
    _source_spec, target, _source, _old, _fresh, heldout = conflict_learning()
    return case, target, heldout, good, cold.library


def _run_arm(case, target, heldout, source_id, library, arm):
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import ResearchConfig, SearchBudget
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_revisits import offer_successor_knowledge

    config = ResearchConfig(
        observation_budget=33,
        compute_budget=20_000,
        minimum_improvement=1e-7,
        max_rounds=1,
        active_probes=False,
        risk_learning=False,
        seed=1237,
        max_candidates=2,
    )
    previous = deepcopy(case["previous"])
    previous["task_id"] = f"unseen_eval_{arm}"
    laboratory = type(case["lab"])(target)
    agent = UnifiedResearchAgent(config, library=library, measurement_budget=0)
    offer = offer_successor_knowledge(
        agent, laboratory, previous, {source_id}, SearchBudget(4_000))
    if offer is None:
        raise RuntimeError(f"missing real {arm} successor offer")
    report = agent.run(
        laboratory, previous=previous,
        continuation_candidate=offer["candidate"])
    program = Program.from_dict(report["program"])
    return {
        "arm": arm,
        "source_id": source_id,
        "offer": {
            "old_evidence_gain": offer["old_evidence_gain"],
            "relative_gain": offer["relative_gain"],
            "minimum_resources": dict(offer["minimum_resources"]),
        },
        "heldout_loss": loss(program, heldout, report["fallback"], .5),
        "report_status": report["status"],
        "visit_costs": dict(report["visit_costs"]),
        "knowledge_returns": [
            {key: row[key] for key in ("source_id", "paired_gain", "retained")}
            for row in report["knowledge_return"]["returns"]
        ],
        "external_model_calls": 0,
    }


def main():
    case, target, heldout, good, cold_library = _fixture()
    results = []
    for arm, source_id in (("bad", case["source_id"]), ("good", good)):
        library_payload = json.loads(json.dumps(cold_library.to_dict()))
        from hegel_machine.cognition.memory import KnowledgeLibrary
        library = KnowledgeLibrary.from_dict(library_payload)
        results.append(_run_arm(case, target, heldout, source_id, library, arm))

    costs = [row["visit_costs"] for row in results]
    assert costs[0] == costs[1], f"arms did not use equal costs: {costs}"
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "protocol": {
            "same_target_prefix": True,
            "shared_source_library_payload": True,
            "same_observation_budget": 33,
            "same_compute_budget": 20_000,
            "heldout_rows": len(heldout),
            "heldout_used_after_run": True,
            "external_model_calls": 0,
        },
        "arms": results,
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
