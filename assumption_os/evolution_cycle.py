"""One-command self-evolution cycle for the Assumption Graph.

The cycle is intentionally conservative.  By default it plans the whole loop
without mutating the graph:

evaluation writeback preview -> conditioned gate -> lifecycle actions ->
candidate proposals -> candidate preflight -> regression/policy plan.

Graph writes require explicit flags.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .candidate_acceptance import apply_accepted_candidates, build_acceptance_payload
from .candidate_eval import build_candidate_eval_payload
from .bayesian_policy import build_bayesian_policy_payload
from .conditioned_eval import GateThresholds, build_conditioned_rows, evaluate_graph_nodes
from .falsification import build_falsification_payload
from .formal_mapping import build_formal_mapping_gate_payload, build_formal_mapping_payload
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .lifecycle import build_lifecycle_payload
from .proposal_overlay import parse_csv_set
from .proposals import build_proposal_payload
from .record_phase2_eval import record_phase2_eval


def build_evolution_cycle_payload(
    *,
    root: Path,
    graph_dir: Path,
    sample_path: Path,
    meta_path: Path,
    judgment_paths: Iterable[Path],
    intervention_variant: str,
    baseline_variant: str,
    eval_id: str,
    top_k: int = 8,
    policy_rerank: bool = False,
    skip_domains: set[str] | None = None,
    skip_missing_meta: bool = True,
    writeback: bool = False,
    conditioned_top_n: int = 25,
    lifecycle_top_n: int | None = None,
    proposal_top_n: int | None = None,
    min_benefit_n: int = 3,
    min_harm_n: int = 3,
    force_proposal_route: bool = True,
    proposals_arg: str = "phase four/assumption_graph/evolution_cycle_proposals.json",
    candidate_judgment_paths: Iterable[Path] | None = None,
    candidate_variant: str | None = None,
    candidate_baseline_variant: str | None = None,
    apply_accepted: bool = False,
) -> dict:
    """Run one self-evolution planning cycle and return an audit payload."""

    judgment_paths = list(judgment_paths)
    candidate_judgment_paths = list(candidate_judgment_paths or [])
    skip_domains = skip_domains or set()

    writeback_summary = record_phase2_eval(
        root=root,
        graph_dir=graph_dir,
        sample_path=sample_path,
        meta_path=meta_path,
        judgment_paths=judgment_paths,
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
        eval_id=eval_id,
        top_k=top_k,
        policy_rerank=policy_rerank,
        skip_domains=skip_domains,
        skip_missing_meta=skip_missing_meta,
        dry_run=not writeback,
    )

    graph = SimpleAssumptionGraph(JsonlGraphStore(graph_dir))
    sample = _load_json(sample_path)
    meta_by_pid = _load_json(meta_path)
    rows = build_conditioned_rows(
        graph=graph,
        sample=sample,
        meta_by_pid=meta_by_pid,
        judgment_paths=judgment_paths,
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
        top_k=top_k,
        policy_rerank=policy_rerank,
        skip_domains=skip_domains,
        skip_missing_meta=skip_missing_meta,
    )
    thresholds = GateThresholds(min_benefit_n=min_benefit_n, min_harm_n=min_harm_n)
    summaries = evaluate_graph_nodes(graph, rows, thresholds=thresholds)[:conditioned_top_n]
    conditioned_payload = {
        "rows": len(rows),
        "thresholds": asdict(thresholds),
        "decision_counts": dict(Counter(s.decision.value for s in summaries)),
        "summaries": [s.to_dict() for s in summaries],
    }
    lifecycle_payload = build_lifecycle_payload(
        conditioned_payload,
        eval_id=f"{eval_id}_lifecycle",
        max_actions=lifecycle_top_n,
    )
    formal_mapping_payload = build_formal_mapping_payload(graph.store)
    proposal_payload = build_proposal_payload(
        graph=graph,
        lifecycle_payload=lifecycle_payload,
        eval_id=f"{eval_id}_proposals",
        max_proposals=proposal_top_n,
    )
    formal_mapping_gate_payload = build_formal_mapping_gate_payload(
        proposal_payload=proposal_payload,
        formal_mapping_payload=formal_mapping_payload,
    )
    preflight_payload = build_candidate_eval_payload(
        graph_dir=graph_dir,
        proposal_payload=proposal_payload,
        sample=sample,
        meta_by_pid=meta_by_pid,
        eval_id=f"{eval_id}_candidate_preflight",
        top_k=top_k,
        policy_rerank=policy_rerank,
        skip_domains=skip_domains,
        skip_missing_meta=skip_missing_meta,
        min_trigger_n=min_benefit_n,
        min_active_trigger_n=min_benefit_n,
        force_proposal_route=force_proposal_route,
        proposals_arg=proposals_arg,
    )

    acceptance_payload = None
    applied_candidate_node_ids: list[str] = []
    if candidate_judgment_paths and candidate_variant and candidate_baseline_variant:
        acceptance_payload = build_acceptance_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            judgment_paths=candidate_judgment_paths,
            candidate_variant=candidate_variant,
            baseline_variant=candidate_baseline_variant,
            eval_id=f"{eval_id}_candidate_acceptance",
        )
        if apply_accepted:
            gated_acceptance_payload = _filter_acceptance_for_formal_mapping_gate(
                acceptance_payload,
                formal_mapping_gate_payload,
            )
            applied_candidate_node_ids = apply_accepted_candidates(
                JsonlGraphStore(graph_dir),
                proposal_payload,
                gated_acceptance_payload,
            )

    regression_predictions = predict_candidate_regressions(preflight_payload)
    falsification_payload = build_falsification_payload(
        proposal_payload=proposal_payload,
        preflight_payload=preflight_payload,
        acceptance_payload=acceptance_payload,
    )
    bayesian_policy_payload = build_bayesian_policy_payload(
        store=JsonlGraphStore(graph_dir),
        proposal_payload=proposal_payload,
        preflight_payload=preflight_payload,
        falsification_payload=falsification_payload,
        acceptance_payload=acceptance_payload,
        regression_predictions=regression_predictions,
    )
    policy_update_plan = build_policy_update_plan(
        proposal_payload=proposal_payload,
        preflight_payload=preflight_payload,
        acceptance_payload=acceptance_payload,
        apply_accepted=apply_accepted,
        applied_candidate_node_ids=applied_candidate_node_ids,
        bayesian_policy_payload=bayesian_policy_payload,
        formal_mapping_gate_payload=formal_mapping_gate_payload,
    )

    return {
        "eval_id": eval_id,
        "mode": {
            "writeback": writeback,
            "apply_accepted": apply_accepted,
            "policy_rerank": policy_rerank,
            "skip_domains": sorted(skip_domains),
            "skip_missing_meta": skip_missing_meta,
            "force_proposal_route": force_proposal_route,
            "proposals_arg": proposals_arg,
        },
        "source": {
            "graph_dir": str(graph_dir),
            "sample": str(sample_path),
            "meta": str(meta_path),
            "judgments": [str(p) for p in judgment_paths],
            "intervention_variant": intervention_variant,
            "baseline_variant": baseline_variant,
        },
        "writeback_summary": writeback_summary,
        "conditioned": conditioned_payload,
        "lifecycle": lifecycle_payload,
        "formal_mapping_audit": formal_mapping_payload,
        "formal_mapping_gate": formal_mapping_gate_payload,
        "proposals": proposal_payload,
        "candidate_preflight": preflight_payload,
        "candidate_acceptance": acceptance_payload,
        "falsification_gate": falsification_payload,
        "regression_predictions": regression_predictions,
        "bayesian_policy": bayesian_policy_payload,
        "policy_update_plan": policy_update_plan,
    }


def predict_candidate_regressions(preflight_payload: dict) -> list[dict]:
    predictions = []
    for summary in preflight_payload.get("summaries", []):
        outside_active = summary.get("outside_active_problem_ids", [])
        control_ids = summary.get("control_problem_ids", [])
        missed = summary.get("missed_trigger_problem_ids", [])
        readiness = summary.get("readiness")
        risk = "low"
        reasons = []
        if outside_active:
            risk = "high"
            reasons.append("candidate or parent is active outside its routed trigger subset")
        if readiness == "needs_scope_fix":
            risk = "high"
            reasons.append("preflight detected active no-fire exposure")
        if readiness == "needs_retrieval_fix":
            risk = "medium" if risk == "low" else risk
            reasons.append("candidate misses routed trigger rows and may underperform")
        if missed and not reasons:
            risk = "medium"
            reasons.append("some trigger rows are still missed")
        if not reasons:
            reasons.append("no preflight regression signal; fresh ablation still required")
        predictions.append({
            "proposal_id": summary.get("proposal_id"),
            "candidate_node_id": summary.get("candidate_node_id"),
            "risk": risk,
            "readiness": readiness,
            "outside_active_problem_ids": outside_active,
            "control_problem_ids": control_ids,
            "reasons": reasons,
        })
    return predictions


def build_policy_update_plan(
    *,
    proposal_payload: dict,
    preflight_payload: dict,
    acceptance_payload: dict | None = None,
    apply_accepted: bool = False,
    applied_candidate_node_ids: list[str] | None = None,
    bayesian_policy_payload: dict | None = None,
    formal_mapping_gate_payload: dict | None = None,
) -> dict:
    accepted = set((acceptance_payload or {}).get("accepted_proposal_ids", []))
    summary_by_proposal = {s["proposal_id"]: s for s in preflight_payload.get("summaries", [])}
    bayes_by_proposal = {
        s["proposal_id"]: s
        for s in (bayesian_policy_payload or {}).get("scores", [])
    }
    formal_gate_by_proposal = {
        g["proposal_id"]: g
        for g in (formal_mapping_gate_payload or {}).get("gates", [])
    }
    actions = []
    for proposal in proposal_payload.get("proposals", []):
        pid = proposal.get("proposal_id")
        preflight = summary_by_proposal.get(pid, {})
        candidate = proposal.get("candidate_node") or {}
        formal_gate = formal_gate_by_proposal.get(pid, {})
        if pid in accepted:
            action = "applied_to_graph" if apply_accepted else "ready_to_apply_with_apply_accepted"
        elif preflight.get("readiness") == "ready_for_fresh_ablation":
            action = "run_fresh_ablation_before_promotion"
        elif preflight.get("readiness") == "needs_scope_fix":
            action = "block_default_activation_and_narrow_scope"
        elif preflight.get("readiness") == "needs_retrieval_fix":
            action = "repair_retrieval_before_ablation"
        elif preflight.get("readiness") == "manifest_only":
            action = "record_manifest_only_no_graph_policy_change"
        else:
            action = "collect_more_evidence"
        action = _apply_formal_mapping_policy_gate(action, proposal, formal_gate)
        actions.append({
            "proposal_id": pid,
            "proposal_type": proposal.get("proposal_type"),
            "parent_node_id": proposal.get("parent_node_id"),
            "candidate_node_id": candidate.get("id"),
            "preflight_readiness": preflight.get("readiness"),
            "formal_mapping_gate": formal_gate or {"decision": "not_applicable"},
            "bayesian_action": bayes_by_proposal.get(pid, {}).get("recommended_action"),
            "bayesian_priority": bayes_by_proposal.get(pid, {}).get("posterior_priority"),
            "bayesian_expected_value": bayes_by_proposal.get(pid, {}).get("expected_value"),
            "policy_action": action,
            "command_hint": preflight.get("command_hint", ""),
        })
    return {
        "accepted_proposal_ids": sorted(accepted),
        "applied_candidate_node_ids": applied_candidate_node_ids or [],
        "actions": actions,
    }


def _filter_acceptance_for_formal_mapping_gate(acceptance_payload: dict, formal_mapping_gate_payload: dict) -> dict:
    blocked = set(formal_mapping_gate_payload.get("blocked_proposal_ids", []))
    if not blocked:
        return acceptance_payload
    out = dict(acceptance_payload)
    out["accepted_proposal_ids"] = [
        pid for pid in acceptance_payload.get("accepted_proposal_ids", []) if pid not in blocked
    ]
    out["formal_mapping_blocked_accepted_proposal_ids"] = sorted(
        pid for pid in acceptance_payload.get("accepted_proposal_ids", []) if pid in blocked
    )
    return out


def _apply_formal_mapping_policy_gate(action: str, proposal: dict, formal_gate: dict) -> str:
    if not formal_gate.get("blocks_policy_update"):
        return action
    if not _is_promotion_sensitive_action(action, proposal):
        return action
    if formal_gate.get("decision") == "block_unsafe_mapping":
        return "block_unsafe_formal_mapping"
    return "repair_formal_mapping_before_policy_update"


def _is_promotion_sensitive_action(action: str, proposal: dict) -> bool:
    if proposal.get("proposal_type") == "promotion_record":
        return True
    return action in {
        "applied_to_graph",
        "ready_to_apply_with_apply_accepted",
        "run_fresh_ablation_before_promotion",
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--judgments", nargs="+", required=True)
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--policy-rerank", action="store_true")
    ap.add_argument("--assumption-graph-skip-domains", default="")
    ap.add_argument("--include-missing-meta", action="store_true")
    ap.add_argument("--writeback", action="store_true")
    ap.add_argument("--conditioned-top-n", type=int, default=25)
    ap.add_argument("--lifecycle-top-n", type=int, default=None)
    ap.add_argument("--proposal-top-n", type=int, default=None)
    ap.add_argument("--min-benefit-n", type=int, default=3)
    ap.add_argument("--min-harm-n", type=int, default=3)
    ap.add_argument("--no-force-proposal-route", action="store_true")
    ap.add_argument("--proposal-artifact-out", default="phase four/assumption_graph/evolution_cycle_proposals.json",
                    help="path used for generated proposal payload and command hints")
    ap.add_argument("--candidate-judgments", nargs="*", default=None)
    ap.add_argument("--candidate-variant", default=None)
    ap.add_argument("--candidate-baseline", default=None)
    ap.add_argument("--apply-accepted", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_evolution_cycle_payload(
        root=root,
        graph_dir=_resolve(root, args.graph_dir),
        sample_path=_resolve(root, args.sample),
        meta_path=_resolve(root, args.meta),
        judgment_paths=[_resolve(root, p) for p in args.judgments],
        intervention_variant=args.intervention,
        baseline_variant=args.baseline,
        eval_id=args.eval_id,
        top_k=args.top_k,
        policy_rerank=args.policy_rerank,
        skip_domains=parse_csv_set(args.assumption_graph_skip_domains),
        skip_missing_meta=not args.include_missing_meta,
        writeback=args.writeback,
        conditioned_top_n=args.conditioned_top_n,
        lifecycle_top_n=args.lifecycle_top_n,
        proposal_top_n=args.proposal_top_n,
        min_benefit_n=args.min_benefit_n,
        min_harm_n=args.min_harm_n,
        force_proposal_route=not args.no_force_proposal_route,
        proposals_arg=args.proposal_artifact_out,
        candidate_judgment_paths=[_resolve(root, p) for p in args.candidate_judgments or []],
        candidate_variant=args.candidate_variant,
        candidate_baseline_variant=args.candidate_baseline,
        apply_accepted=args.apply_accepted,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.proposal_artifact_out:
        proposal_out = _resolve(root, args.proposal_artifact_out)
        proposal_out.parent.mkdir(parents=True, exist_ok=True)
        proposal_out.write_text(json.dumps(payload["proposals"], ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
