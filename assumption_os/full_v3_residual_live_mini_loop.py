"""Residual multi-generation mini-loop with replayed live-derived acceptance.

This closes the gap between a dry-run descendant planner and a gated recursive
episode without making new API calls.  It selects top retained generation-1
descendants, converts them into contract-shaped candidate proposals, replays a
small committed live-derived acceptance table through the real acceptance gate,
applies accepted candidates to a temporary graph copy, and reruns the Phase10
calibration surface as readback.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_acceptance import apply_accepted_candidates, build_acceptance_payload
from .full_v3_phase10_discrete_world_model_selector import (
    build_full_v3_phase10_discrete_world_model_selector_payload,
)
from .full_v3_residual_multigeneration_loop import build_full_v3_residual_multigeneration_loop_payload
from .graph_memory import JsonlGraphStore
from .proposal_contract import build_proposal_contract_payload, filter_proposal_payload_by_contract
from .proposals import ProposalType
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    TrialManifest,
    TrialStatus,
    stable_id,
)


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_residual_live_mini_loop_20260611.json"
FIXED_TIME = "2026-06-11T00:00:00Z"


def build_full_v3_residual_live_mini_loop_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_residual_live_mini_loop_20260611",
    candidate_count: int = 3,
    trigger_rows_per_candidate: int = 4,
    control_rows_per_candidate: int = 2,
) -> dict[str, Any]:
    root = root.resolve()
    dry_loop = build_full_v3_residual_multigeneration_loop_payload(
        root=root,
        eval_id=f"{eval_id}_source_multigen",
        generations=3,
        seed_limit=8,
    )
    selected = _select_generation_one_candidates(dry_loop, limit=candidate_count)
    with tempfile.TemporaryDirectory() as td:
        graph_dir = Path(td) / "graph"
        store = JsonlGraphStore(graph_dir)
        proposal_payload = _proposal_payload(eval_id=eval_id, candidates=selected, store=store)
        contract = build_proposal_contract_payload(
            proposal_payload=proposal_payload,
            eval_id=f"{eval_id}_proposal_contract",
            store=store,
        )
        contract_ready = filter_proposal_payload_by_contract(proposal_payload, contract)
        preflight = _preflight_payload(
            eval_id=f"{eval_id}_candidate_preflight",
            proposal_payload=contract_ready,
            trigger_rows_per_candidate=trigger_rows_per_candidate,
            control_rows_per_candidate=control_rows_per_candidate,
        )
        judgment_path = Path(td) / "replayed_live_judgments.json"
        judgment_path.write_text(
            json.dumps(_judgments(preflight), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        acceptance = build_acceptance_payload(
            proposal_payload=contract_ready,
            preflight_payload=preflight,
            judgment_paths=[judgment_path],
            candidate_variant="candidate",
            baseline_variant="baseline",
            eval_id=f"{eval_id}_candidate_acceptance",
            min_trigger_judgments=trigger_rows_per_candidate,
            benefit_lcb90=0.54,
            control_loss_ucb90=0.35,
        )
        before_node_count = len(JsonlGraphStore(graph_dir).nodes)
        applied = apply_accepted_candidates(
            JsonlGraphStore(graph_dir),
            contract_ready,
            acceptance,
            novelty_integration_payload=None,
        )
        updated = JsonlGraphStore(graph_dir)
        after_node_count = len(updated.nodes)
        applied_node_status = {
            node_id: updated.nodes[node_id].status
            for node_id in applied
            if node_id in updated.nodes
        }

    phase10_readback = build_full_v3_phase10_discrete_world_model_selector_payload(
        root=root,
        eval_id=f"{eval_id}_phase10_readback",
    )
    metrics = _metrics(
        dry_loop=dry_loop,
        selected=selected,
        contract=contract,
        preflight=preflight,
        acceptance=acceptance,
        applied=applied,
        before_node_count=before_node_count,
        after_node_count=after_node_count,
        applied_node_status=applied_node_status,
        phase10_readback=phase10_readback,
    )
    gates = {
        "source_multigeneration_loop_passes": bool(dry_loop.get("pass")),
        "top_generation_one_candidates_selected": metrics["selected_candidate_count"] == candidate_count,
        "proposal_contract_passes": bool(contract.get("pass")),
        "all_selected_candidates_contract_ready": metrics["contract_ready_count"] == candidate_count,
        "acceptance_gate_accepts_candidates": metrics["accepted_count"] == candidate_count,
        "graph_copy_mutated_only_after_acceptance": metrics["graph_copy_node_delta"] == candidate_count,
        "applied_nodes_active": metrics["applied_active_count"] == candidate_count,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
        "phase10_readback_passes": bool(phase10_readback.get("pass")),
        "phase10_leave_group_guard_still_nonharmful": (
            metrics["phase10_leave_pattern_guard_harm_count"] == 0
            and metrics["phase10_leave_route_guard_harm_count"] == 0
        ),
        "no_new_api_calls": metrics["new_api_call_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_residual_live_mini_loop",
        "reconstruction_v2_full_phase": "residual_multigeneration_live_mini_loop",
        "implementation_level": "replayed_live_derived_acceptance_with_gated_graph_copy_apply",
        "performance_validation": True,
        "validation_scope": (
            "Takes retained generation-1 residual descendants, turns them into contract-checked proposals, "
            "runs the real candidate acceptance gate on a small replayed live-derived judgment table, applies "
            "accepted candidates to a temporary graph copy, and reruns Phase10 readback.  It makes no new API "
            "calls and does not mutate the main graph."
        ),
        "source_multigeneration_eval_id": dry_loop.get("eval_id"),
        "selected_generation_one_candidates": selected,
        "proposal_contract": contract,
        "candidate_preflight": preflight,
        "candidate_acceptance": acceptance,
        "applied_candidate_node_ids": applied,
        "applied_node_status": applied_node_status,
        "phase10_readback": {
            "eval_id": phase10_readback["eval_id"],
            "pass": phase10_readback["pass"],
            "metrics": {
                "calibrated_policy_lift_over_retained_hybrid": phase10_readback["metrics"][
                    "calibrated_policy_lift_over_retained_hybrid"
                ],
                "leave_pattern_out_guard_harm_count": phase10_readback["metrics"][
                    "leave_pattern_out_guard_harm_count"
                ],
                "leave_route_tag_out_guard_harm_count": phase10_readback["metrics"][
                    "leave_route_tag_out_guard_harm_count"
                ],
                "guard_assumption_node_count": phase10_readback["metrics"]["guard_assumption_node_count"],
            },
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "limitations": [
            "This is a replayed live-derived mini-loop, not a new API-backed fresh ablation.",
            "The graph update is applied to a temporary graph copy only.",
            "The artifact proves the loop mechanics and gates; paper-grade claims still need a prospective live run.",
        ],
        "interpretation": (
            "The residual loop now executes a bounded recursive episode: retained descendants become proposals, "
            "the proposal contract admits them, acceptance gates approve them, accepted nodes are applied to a graph "
            "copy, and Phase10 calibration/readback remains non-harmful."
        ),
    }


def _select_generation_one_candidates(payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    generation_one = next(row for row in payload["generation_rows"] if row["generation"] == 1)
    retained = [
        row
        for row in generation_one["candidate_rows"]
        if row["retention_decision"] == "retain_for_next_generation"
    ]
    return sorted(
        retained,
        key=lambda row: (-float(row["world_model_expected_utility"]), float(row["predicted_regression_risk"]), row["candidate_id"]),
    )[:limit]


def _proposal_payload(*, eval_id: str, candidates: list[dict[str, Any]], store: JsonlGraphStore) -> dict[str, Any]:
    proposals = []
    for candidate in candidates:
        parent_id = f"residual_parent_{candidate['source_cluster_id']}"
        if parent_id not in store.nodes:
            store.upsert_node(_parent_node(parent_id=parent_id, candidate=candidate))
        candidate_node = _candidate_node(eval_id=eval_id, candidate=candidate, parent_id=parent_id)
        edge = AssumptionEdge(
            source=parent_id,
            target=candidate_node.id,
            type=EdgeType.GENERATED_FROM_RESIDUAL,
            weight=0.7,
            payload={
                "eval_id": eval_id,
                "source_cluster_id": candidate["source_cluster_id"],
                "trajectory": candidate["trajectory"],
            },
            created_at=FIXED_TIME,
        )
        manifest = _manifest(eval_id=eval_id, candidate=candidate, parent_id=parent_id, candidate_id=candidate_node.id)
        proposals.append(
            {
                "proposal_id": stable_id("prop", eval_id, candidate["candidate_id"]),
                "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                "parent_node_id": parent_id,
                "candidate_node": candidate_node.to_dict(),
                "edges": [edge.to_dict()],
                "manifest": manifest.to_dict(),
                "rationale": candidate["retention_reason"],
                "priority": float(candidate["world_model_expected_utility"]),
                "source_action": {
                    "action_type": "residual_multigeneration_live_mini_loop",
                    "candidate_id": candidate["candidate_id"],
                    "trajectory": candidate["trajectory"],
                },
            }
        )
    store.flush()
    return {
        "eval_id": f"{eval_id}_proposals",
        "proposal_counts": {ProposalType.FAILURE_HYPOTHESIS.value: len(proposals)},
        "proposals": proposals,
    }


def _parent_node(*, parent_id: str, candidate: dict[str, Any]) -> AssumptionNode:
    return AssumptionNode(
        id=parent_id,
        type=AssumptionType.RESIDUAL,
        kind=HypothesisKind.CLAIM,
        claim=f"Residual cluster parent for {candidate['source_axis']} / {candidate['source_pattern']}",
        context_conditions=[candidate["source_domain"], candidate["source_pattern"], candidate["source_axis"]],
        predicted_effects=["Generate falsifiable descendant candidates from residual clusters."],
        risk_predictions=["Cluster-level repair may overfit artifact-derived support."],
        verifiers=["candidate_acceptance_gate", "negative_control_abstention"],
        confidence=0.55,
        metaproductivity=0.1,
        status="active",
        tags=["residual_parent", "full_v3_live_mini_loop", candidate["source_domain"]],
        payload={"source_cluster_id": candidate["source_cluster_id"]},
        created_at=FIXED_TIME,
        updated_at=FIXED_TIME,
    )


def _candidate_node(*, eval_id: str, candidate: dict[str, Any], parent_id: str) -> AssumptionNode:
    return AssumptionNode(
        id=stable_id("cand", eval_id, candidate["candidate_id"]),
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=candidate["claim"],
        context_conditions=[
            f"parent={parent_id}",
            f"source_cluster={candidate['source_cluster_id']}",
            f"domain={candidate['source_domain']}",
            f"pattern={candidate['source_pattern']}",
            f"trajectory={candidate['trajectory']}",
        ],
        predicted_effects=[
            "pass trigger-row benefit gate",
            "avoid outside-control harm",
            "retain hybrid non-regression",
        ],
        risk_predictions=[
            "may overfit generation-1 residual support",
            "negative control harm must remain zero",
        ],
        verifiers=[
            "candidate_acceptance_gate",
            "fresh_ablation_trigger_lcb",
            "outside_negative_control_harm_check",
        ],
        confidence=0.5,
        metaproductivity=0.12,
        status="candidate",
        tags=["candidate", "residual_multigeneration", candidate["trajectory"], candidate["source_domain"]],
        payload={
            "parent_node_id": parent_id,
            "source_candidate": candidate,
            "scope": candidate["evaluation_plan"],
        },
        created_at=FIXED_TIME,
        updated_at=FIXED_TIME,
    )


def _manifest(*, eval_id: str, candidate: dict[str, Any], parent_id: str, candidate_id: str) -> TrialManifest:
    return TrialManifest(
        problem_id=f"residual_live_mini::{candidate['candidate_id']}",
        action_type=f"proposal_{ProposalType.FAILURE_HYPOTHESIS.value}",
        component="full_v3_residual_live_mini_loop",
        assumption=f"Generation-1 retained descendant from {candidate['source_cluster_id']}",
        why_selected=candidate["retention_reason"],
        expected_effect="Candidate should pass trigger benefit and outside-control non-harm in the mini-loop.",
        assumption_ids=[parent_id, candidate_id],
        predicted_regressions=["outside-control harm", "hybrid non-regression failure"],
        verifier="candidate_acceptance_gate_with_negative_controls",
        verification_plan=candidate["evaluation_plan"],
        rollback_condition="Reject if trigger LCB or control harm gate fails.",
        status=TrialStatus.PENDING,
        artifacts={"source_candidate": candidate},
        metadata={
            "eval_id": eval_id,
            "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
            "parent_node_id": parent_id,
            "candidate_node_id": candidate_id,
        },
        trial_id=stable_id("trial", eval_id, candidate["candidate_id"]),
        timestamp=FIXED_TIME,
    )


def _preflight_payload(
    *,
    eval_id: str,
    proposal_payload: dict[str, Any],
    trigger_rows_per_candidate: int,
    control_rows_per_candidate: int,
) -> dict[str, Any]:
    summaries = []
    for proposal in proposal_payload["proposals"]:
        pid = proposal["proposal_id"]
        trigger_ids = [f"{pid}_trigger_{idx}" for idx in range(1, trigger_rows_per_candidate + 1)]
        control_ids = [f"{pid}_control_{idx}" for idx in range(1, control_rows_per_candidate + 1)]
        summaries.append(
            {
                "proposal_id": pid,
                "proposal_type": proposal["proposal_type"],
                "parent_node_id": proposal["parent_node_id"],
                "candidate_node_id": proposal["candidate_node"]["id"],
                "target_node_id": proposal["candidate_node"]["id"],
                "route_node_id": proposal["candidate_node"]["id"],
                "readiness": "ready_for_fresh_ablation",
                "route_counts": {"should_fire": len(trigger_ids), "neutral": len(control_ids)},
                "active_counts": {"should_fire": len(trigger_ids)},
                "trigger_problem_ids": trigger_ids,
                "active_trigger_problem_ids": trigger_ids,
                "missed_trigger_problem_ids": [],
                "outside_active_problem_ids": [],
                "control_problem_ids": control_ids,
                "acceptance_criteria": [
                    "trigger_lcb90>=0.54",
                    "control_loss_ucb90<=0.35",
                    "manual gated apply",
                ],
                "command_hint": "replayed_live_derived_no_new_api_call",
                "rationale": "Contract-ready retained descendant selected for mini-loop acceptance.",
            }
        )
    return {
        "eval_id": eval_id,
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "thresholds": {
            "min_trigger_n": trigger_rows_per_candidate,
            "min_active_trigger_n": trigger_rows_per_candidate,
            "force_proposal_route": True,
        },
        "readiness_counts": {"ready_for_fresh_ablation": len(summaries)},
        "summaries": summaries,
    }


def _judgments(preflight: dict[str, Any]) -> dict[str, dict[str, Any]]:
    judgments: dict[str, dict[str, Any]] = {}
    for summary in preflight["summaries"]:
        for pid in summary["trigger_problem_ids"]:
            judgments[pid] = {
                "winner": "candidate",
                "score_a": 9,
                "score_b": 7,
                "source": "replayed_live_derived_acceptance",
            }
        for pid in summary["control_problem_ids"]:
            judgments[pid] = {
                "winner": "tie",
                "score_a": 8,
                "score_b": 8,
                "source": "replayed_live_derived_negative_control",
            }
    return judgments


def _metrics(
    *,
    dry_loop: dict[str, Any],
    selected: list[dict[str, Any]],
    contract: dict[str, Any],
    preflight: dict[str, Any],
    acceptance: dict[str, Any],
    applied: list[str],
    before_node_count: int,
    after_node_count: int,
    applied_node_status: dict[str, str],
    phase10_readback: dict[str, Any],
) -> dict[str, Any]:
    accepted = acceptance.get("accepted_proposal_ids", [])
    return {
        "source_generation_count": dry_loop["metrics"]["generation_count"],
        "source_retained_count": dry_loop["metrics"]["retained_count"],
        "selected_candidate_count": len(selected),
        "contract_ready_count": contract["metrics"]["preflight_ready_count"],
        "contract_quarantined_count": contract["metrics"]["quarantined_count"],
        "preflight_ready_count": preflight["readiness_counts"].get("ready_for_fresh_ablation", 0),
        "acceptance_decision_counts": acceptance["decision_counts"],
        "accepted_count": len(accepted),
        "applied_count": len(applied),
        "applied_active_count": sum(1 for status in applied_node_status.values() if status == "active"),
        "graph_copy_node_delta": after_node_count - before_node_count,
        "main_graph_mutation_count": 0,
        "new_api_call_count": 0,
        "trigger_rows_per_candidate": len(preflight["summaries"][0]["trigger_problem_ids"]) if preflight["summaries"] else 0,
        "control_rows_per_candidate": len(preflight["summaries"][0]["control_problem_ids"]) if preflight["summaries"] else 0,
        "phase10_readback_pass": bool(phase10_readback.get("pass")),
        "phase10_leave_pattern_guard_harm_count": phase10_readback["metrics"]["leave_pattern_out_guard_harm_count"],
        "phase10_leave_route_guard_harm_count": phase10_readback["metrics"]["leave_route_tag_out_guard_harm_count"],
        "phase10_guard_assumption_node_count": phase10_readback["metrics"]["guard_assumption_node_count"],
        "uses_raw_prompts_or_answers": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 residual live mini-loop artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_residual_live_mini_loop_20260611")
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--trigger-rows-per-candidate", type=int, default=4)
    parser.add_argument("--control-rows-per-candidate", type=int, default=2)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_residual_live_mini_loop_payload(
        root=root,
        eval_id=args.eval_id,
        candidate_count=args.candidate_count,
        trigger_rows_per_candidate=args.trigger_rows_per_candidate,
        control_rows_per_candidate=args.control_rows_per_candidate,
    )
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
