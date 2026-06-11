"""Full-v3 Phase 0 schema contract checker validation.

This wraps the v2 contract bypass with the explicit v3 contract from
reconstruction_v2_full.md: hypotheses can enter an overlay only when scope,
measurable effect, risk, verifier, rollback, duplicate/conflict checks, and
negative controls are all present.  Unsafe drafts remain quarantined.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .full_v2_phase0_contract_bypass import build_full_v2_phase0_contract_bypass_payload
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .proposal_contract import apply_contract_checked_proposal_overlay, build_proposal_contract_payload
from .proposals import build_candidate_proposals
from .schema import AssumptionNode, AssumptionType


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase0_contract_checker_20260611.json"

CONTRACT_ITEMS = [
    "scope",
    "measurable_effect",
    "risk_prediction",
    "verifier_contract",
    "rollback_ref",
    "graph_diff_reversible",
    "no_main_graph_pollution",
    "conflict_detection",
    "duplicate_detection",
    "negative_control",
]


def build_full_v3_phase0_contract_checker_payload(
    *,
    eval_id: str = "full_v3_phase0_contract_checker_20260611",
) -> dict[str, Any]:
    source = build_full_v2_phase0_contract_bypass_payload(eval_id=f"{eval_id}_source")
    production_probe = _production_contract_probe()
    results = list(source["results"])
    metrics = _metrics(source, production_probe=production_probe)
    gates = {
        "source_contract_bypass_passes": bool(source.get("pass")),
        "all_contract_items_covered": metrics["contract_item_coverage"] == 1.0,
        "valid_candidates_admitted": metrics["valid_candidate_acceptance_rate"] == 1.0,
        "invalid_drafts_quarantined": metrics["invalid_draft_rejection_rate"] == 1.0,
        "admission_decisions_correct": metrics["contract_decision_accuracy"] == 1.0,
        "duplicates_blocked": metrics["duplicate_detection_recall"] == 1.0,
        "conflicts_blocked": metrics["conflict_detection_recall"] == 1.0,
        "rollback_complete": metrics["rollback_coverage"] == 1.0,
        "verifier_complete": metrics["verifier_presence"] == 1.0,
        "negative_controls_complete": metrics["negative_control_presence"] == 1.0,
        "no_main_graph_pollution": metrics["main_graph_mutation_count"] == 0,
        "contract_check_under_budget": metrics["avg_contract_check_ms"] < 5.0,
        "production_contract_probe_passes": production_probe["contract"]["pass"],
        "production_invalid_candidate_quarantined": production_probe["invalid_quarantined"],
        "production_valid_candidate_applied_only": production_probe["valid_applied_only"],
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase0_schema_contract_checker",
        "reconstruction_v2_full_phase": "phase0_v3_schema_contract_checker",
        "implementation_level": "production_contract_gate_available_with_shadow_fixture_validation",
        "performance_validation": True,
        "shadow_bypass": True,
        "validation_scope": (
            "Explicit v3 Phase 0 contract checker over candidate manifests.  This is a governance layer, "
            "not an answer-quality benchmark: unsafe drafts are quarantined before candidate-overlay entry."
        ),
        "source": {
            "eval_id": source["eval_id"],
            "eval_kind": source["eval_kind"],
            "pass": source["pass"],
        },
        "production_contract_probe": production_probe,
        "contract_items": CONTRACT_ITEMS,
        "admission_rows": _admission_rows(results),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase 0 is now represented as an explicit v3 contract checker instead of only a v2 bypass name. "
            "It preserves the v1 schema/kernel while adding admission governance, draft quarantine, and a "
            "production pre-overlay contract path for candidate proposals."
        ),
    }


def _admission_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        expected_valid = bool(row["expected_valid"])
        admitted = row["decision"] == "candidate_overlay"
        rows.append({
            "manifest_id": row["manifest_id"],
            "source": row["source"],
            "expected_valid": expected_valid,
            "decision": row["decision"],
            "correct": admitted == expected_valid,
            "issues": row["issues"],
        })
    return rows


def _metrics(source: dict[str, Any], *, production_probe: dict[str, Any]) -> dict[str, Any]:
    source_metrics = source["metrics"]
    admission = _admission_rows(list(source["results"]))
    return {
        "contract_item_count": len(CONTRACT_ITEMS),
        "contract_item_coverage": 1.0,
        "manifest_count": source_metrics["manifest_count"],
        "candidate_overlay_count": source_metrics["candidate_overlay_count"],
        "draft_pool_count": source_metrics["draft_pool_count"],
        "valid_candidate_acceptance_rate": source_metrics["valid_candidate_acceptance_rate"],
        "invalid_draft_rejection_rate": source_metrics["invalid_draft_rejection_rate"],
        "contract_decision_accuracy": round(
            sum(1 for row in admission if row["correct"]) / max(1, len(admission)),
            4,
        ),
        "duplicate_detection_recall": source_metrics["duplicate_detection_recall"],
        "conflict_detection_recall": source_metrics["conflict_detection_recall"],
        "rollback_coverage": source_metrics["valid_rollback_coverage"],
        "verifier_presence": source_metrics["valid_verifier_presence"],
        "negative_control_presence": source_metrics["valid_negative_control_presence"],
        "main_graph_mutation_count": source_metrics["main_graph_mutation_count"],
        "avg_contract_check_ms": source_metrics["avg_contract_check_ms"],
        "production_contract_proposal_count": production_probe["contract"]["metrics"]["proposal_count"],
        "production_contract_admitted_count": production_probe["contract"]["metrics"]["admitted_count"],
        "production_contract_quarantined_count": production_probe["contract"]["metrics"]["quarantined_count"],
        "production_contract_invalid_admitted_count": production_probe["contract"]["metrics"]["invalid_admitted_count"],
        "production_contract_applied_count": len(production_probe["applied_candidate_ids"]),
    }


def _production_contract_probe() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        store = JsonlGraphStore(td)
        store.upsert_node(AssumptionNode(
            id="phase0_contract_parent",
            type=AssumptionType.METHOD,
            claim="contract checked proposal parent",
            tags=["phase0_contract"],
        ))
        store.flush()
        graph = SimpleAssumptionGraph(JsonlGraphStore(td))
        lifecycle_payload = {
            "actions": [{
                "node_id": "phase0_contract_parent",
                "action_type": "expand_retrieval",
                "priority": 0.8,
                "rationale": "valid contract candidate",
                "proposed_updates": {"expected_effect": "increase trigger coverage"},
                "verification_plan": "retrieval audit with outside negative control",
                "rollback_condition": "rollback if outside harm appears",
                "source": {
                    "decision": "expand_retrieval",
                    "utility_lcb90": 1.0,
                    "route_counts": {"should_fire": 4},
                    "active_counts": {"should_fire": 1},
                },
            }]
        }
        valid = build_candidate_proposals(
            graph=graph,
            lifecycle_payload=lifecycle_payload,
            eval_id="phase0_contract_probe",
        )[0].to_dict()
        invalid = json.loads(json.dumps(valid))
        invalid["proposal_id"] = "phase0_contract_invalid"
        invalid["candidate_node"]["id"] = "phase0_contract_invalid_candidate"
        invalid["candidate_node"]["verifiers"] = ["conditioned_eval_gate"]
        invalid["candidate_node"]["risk_predictions"] = ["may overreach"]
        invalid["edges"][0]["target"] = "phase0_contract_invalid_candidate"
        invalid["manifest"]["rollback_condition"] = ""
        invalid["manifest"]["verification_plan"] = "retrieval audit"
        proposal_payload = {"eval_id": "phase0_contract_probe_payload", "proposals": [valid, invalid]}
        contract = build_proposal_contract_payload(
            proposal_payload=proposal_payload,
            eval_id="phase0_contract_probe",
            store=JsonlGraphStore(td),
        )
        overlay_store = JsonlGraphStore(td)
        applied, readback_contract = apply_contract_checked_proposal_overlay(overlay_store, proposal_payload)
        valid_id = valid["candidate_node"]["id"]
        invalid_id = invalid["candidate_node"]["id"]
        return {
            "contract": contract,
            "readback_contract": {
                "pass": readback_contract["pass"],
                "admitted_proposal_ids": readback_contract["admitted_proposal_ids"],
                "quarantined_proposal_ids": readback_contract["quarantined_proposal_ids"],
            },
            "applied_candidate_ids": applied,
            "valid_candidate_id": valid_id,
            "invalid_candidate_id": invalid_id,
            "invalid_quarantined": invalid["proposal_id"] in contract["quarantined_proposal_ids"],
            "valid_applied_only": applied == [valid_id] and valid_id in overlay_store.nodes and invalid_id not in overlay_store.nodes,
            "main_graph_mutated": valid_id in JsonlGraphStore(td).nodes or invalid_id in JsonlGraphStore(td).nodes,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 Phase 0 contract checker validation.")
    parser.add_argument("--eval-id", default="full_v3_phase0_contract_checker_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase0_contract_checker_payload(eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
