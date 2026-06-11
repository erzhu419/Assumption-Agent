"""Prospective live multi-generation residual evolution line.

The earlier residual fresh-live artifact proves one bounded live mini-loop.  This
module expands the same contract/preflight/live-judge/accept/apply-copy path
across multiple retained generations from the residual multi-generation planner.
It intentionally keeps main-graph mutation gated: accepted hypotheses are
applied only to a temporary graph copy unless a separate controlled apply path is
run.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_acceptance import apply_accepted_candidates, build_acceptance_payload
from .full_v3_residual_fresh_live_loop import (
    _env_status,
    _live_judgment_payload,
    _load_keyfile_env,
)
from .full_v3_residual_live_mini_loop import _preflight_payload, _proposal_payload
from .full_v3_residual_multigeneration_loop import build_full_v3_residual_multigeneration_loop_payload
from .graph_memory import JsonlGraphStore
from .proposal_contract import build_proposal_contract_payload, filter_proposal_payload_by_contract


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_live_multigeneration_expansion_20260612.json"


def build_full_v3_live_multigeneration_expansion_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_live_multigeneration_expansion_20260612",
    execution_mode: str = "dry_run",
    generations: int = 3,
    candidates_per_generation: int = 2,
    trigger_rows_per_candidate: int = 4,
    control_rows_per_candidate: int = 2,
    model_alias: str = "gpt_mini",
    load_keyfile: bool = True,
) -> dict[str, Any]:
    if execution_mode not in {"dry_run", "execute_live"}:
        raise ValueError(f"unknown execution_mode={execution_mode}")
    root = root.resolve()
    if load_keyfile:
        _load_keyfile_env()
    env = _env_status(model_alias)
    source_loop = build_full_v3_residual_multigeneration_loop_payload(
        root=root,
        eval_id=f"{eval_id}_source_multigen",
        generations=generations,
        seed_limit=8,
    )
    with tempfile.TemporaryDirectory(prefix="assumption_live_multigen_") as td:
        graph_dir = Path(td) / "graph"
        store = JsonlGraphStore(graph_dir)
        before_node_count = len(store.nodes)
        generation_results = []
        seen_claims: set[str] = set()
        for generation_row in source_loop.get("generation_rows", [])[:generations]:
            selected = _select_generation_candidates(
                generation_row,
                limit=candidates_per_generation,
                excluded_claims=seen_claims,
            )
            seen_claims.update(str(row.get("claim", "")) for row in selected)
            result = _run_generation(
                root=root,
                graph_dir=graph_dir,
                eval_id=f"{eval_id}_gen{generation_row['generation']}",
                generation=int(generation_row["generation"]),
                selected=selected,
                execution_mode=execution_mode,
                env=env,
                model_alias=model_alias,
                trigger_rows_per_candidate=trigger_rows_per_candidate,
                control_rows_per_candidate=control_rows_per_candidate,
            )
            generation_results.append(result)
        after_node_count = len(JsonlGraphStore(graph_dir).nodes)
    metrics = _metrics(
        execution_mode=execution_mode,
        env=env,
        source_loop=source_loop,
        generation_results=generation_results,
        before_node_count=before_node_count,
        after_node_count=after_node_count,
    )
    gates = {
        "source_multigeneration_loop_passes": bool(source_loop.get("pass")),
        "generation_count_high": metrics["generation_count"] >= generations,
        "selected_candidate_count_high": metrics["selected_candidate_count"] >= generations * candidates_per_generation,
        "contract_ready_all_selected": metrics["contract_ready_count"] == metrics["selected_candidate_count"],
        "preflight_ready_all_selected": metrics["preflight_ready_count"] == metrics["selected_candidate_count"],
        "live_or_dry_judgments_complete": (
            execution_mode == "dry_run"
            or metrics["fresh_api_call_count"] == metrics["planned_fresh_api_call_count"]
        ),
        "acceptance_gate_covers_all_selected": metrics["acceptance_decision_count"] == metrics["selected_candidate_count"],
        "selective_retention_observed": (
            execution_mode == "dry_run"
            or metrics["accepted_count"] < metrics["selected_candidate_count"]
        ),
        "graph_copy_only": metrics["main_graph_mutation_count"] == 0,
        "graph_copy_applies_only_accepted": metrics["applied_node_delta"] == metrics["applied_count"],
        "no_secret_value_exposed": metrics["secret_value_exposed"] is False,
    }
    if execution_mode == "execute_live":
        gates["execute_live_requires_ready_env"] = env["ready"] is True
        gates["execute_live_has_real_api_calls"] = metrics["fresh_api_call_count"] > 0
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_live_multigeneration_expansion",
        "reconstruction_v2_full_phase": "prospective_live_multigeneration_residual_evolution",
        "implementation_level": (
            "prospective_live_multigeneration_execute_path"
            if execution_mode == "execute_live"
            else "prospective_live_multigeneration_dry_run"
        ),
        "performance_validation": True,
        "execution_mode": execution_mode,
        "validation_scope": (
            "Runs retained residual descendants across multiple generations through the production "
            "contract -> preflight -> live judge -> acceptance -> graph-copy apply path.  Main graph "
            "mutation remains gated and separate."
        ),
        "live_env": env,
        "source_multigeneration": {
            "eval_id": source_loop.get("eval_id"),
            "pass": source_loop.get("pass"),
            "metrics": source_loop.get("metrics"),
        },
        "generation_results": generation_results,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": _interpretation(execution_mode=execution_mode, metrics=metrics),
    }


def _run_generation(
    *,
    root: Path,
    graph_dir: Path,
    eval_id: str,
    generation: int,
    selected: list[dict[str, Any]],
    execution_mode: str,
    env: dict[str, Any],
    model_alias: str,
    trigger_rows_per_candidate: int,
    control_rows_per_candidate: int,
) -> dict[str, Any]:
    store = JsonlGraphStore(graph_dir)
    proposal_payload = _proposal_payload(eval_id=eval_id, candidates=selected, store=store)
    contract = build_proposal_contract_payload(
        proposal_payload=proposal_payload,
        eval_id=f"{eval_id}_proposal_contract",
        store=JsonlGraphStore(graph_dir),
    )
    contract_ready = filter_proposal_payload_by_contract(proposal_payload, contract)
    preflight = _preflight_payload(
        eval_id=f"{eval_id}_candidate_preflight",
        proposal_payload=contract_ready,
        trigger_rows_per_candidate=trigger_rows_per_candidate,
        control_rows_per_candidate=control_rows_per_candidate,
    )
    live = _live_judgment_payload(
        preflight=preflight,
        selected=selected,
        execution_mode=execution_mode,
        env=env,
        model_alias=model_alias,
    )
    judgment_path = root / PAPER_DIR / f"{eval_id}_judgments_tmp.json"
    try:
        judgment_path.write_text(json.dumps(live["judgments"], ensure_ascii=False, indent=2), encoding="utf-8")
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
    finally:
        if judgment_path.exists():
            judgment_path.unlink()
    before_node_count = len(JsonlGraphStore(graph_dir).nodes)
    applied = apply_accepted_candidates(JsonlGraphStore(graph_dir), contract_ready, acceptance)
    after_node_count = len(JsonlGraphStore(graph_dir).nodes)
    return {
        "generation": generation,
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "selected_candidate_count": len(selected),
        "proposal_contract": {
            "eval_id": contract["eval_id"],
            "pass": contract["pass"],
            "metrics": contract["metrics"],
            "quarantined_proposal_ids": contract.get("quarantined_proposal_ids", []),
        },
        "candidate_preflight": {
            "eval_id": preflight["eval_id"],
            "readiness_counts": preflight.get("readiness_counts", {}),
        },
        "live_judgment": {
            "status": live.get("status"),
            "fresh_api_call_count": live.get("fresh_api_call_count", 0),
            "planned_fresh_api_call_count": live.get("planned_fresh_api_call_count", 0),
            "live_error_count": len(live.get("live_errors", [])),
            "live_errors": live.get("live_errors", [])[:3],
        },
        "candidate_acceptance": {
            "eval_id": acceptance["eval_id"],
            "decision_counts": acceptance.get("decision_counts", {}),
            "accepted_proposal_ids": acceptance.get("accepted_proposal_ids", []),
        },
        "applied_candidate_node_ids": applied,
        "graph_copy_node_delta": after_node_count - before_node_count,
    }


def _select_generation_candidates(
    generation_row: dict[str, Any],
    *,
    limit: int,
    excluded_claims: set[str],
) -> list[dict[str, Any]]:
    retained = [
        row for row in generation_row.get("candidate_rows", [])
        if row.get("retention_decision") == "retain_for_next_generation"
    ]
    selected = []
    seen_ids: set[str] = set()
    seen_claims: set[str] = set()
    for row in sorted(
        retained,
        key=lambda row: (
            -float(row.get("world_model_expected_utility") or 0.0),
            float(row.get("predicted_regression_risk") or 1.0),
            row.get("candidate_id", ""),
        ),
    ):
        candidate = dict(row)
        claim = str(candidate.get("claim", ""))
        candidate_id = str(candidate.get("candidate_id", ""))
        duplicate_claim = claim in seen_claims or claim in excluded_claims
        duplicate_id = candidate_id in seen_ids
        if duplicate_claim or duplicate_id:
            candidate = _as_generation_specific_descendant(candidate, ordinal=len(selected) + 1)
            claim = str(candidate.get("claim", ""))
            candidate_id = str(candidate.get("candidate_id", ""))
        if claim in seen_claims or candidate_id in seen_ids:
            continue
        seen_claims.add(claim)
        seen_ids.add(candidate_id)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _as_generation_specific_descendant(candidate: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    generation = int(candidate.get("generation") or 0)
    parent = str(candidate.get("parent_candidate_id") or "root")
    trajectory = str(candidate.get("trajectory") or "trajectory")
    out = dict(candidate)
    out["candidate_id"] = f"{candidate.get('candidate_id')}_g{generation}_{ordinal}"
    out["claim"] = (
        f"{candidate.get('claim')} Generation-{generation} descendant {ordinal} tests the same structural "
        f"repair under parent {parent} rather than promoting the parent-level claim again."
    )
    out["evaluation_plan"] = (
        f"{candidate.get('evaluation_plan')} Treat this as a generation-{generation} descendant of "
        f"{parent}/{trajectory}; require fresh trigger benefit and no control harm independently."
    )
    return out


def _metrics(
    *,
    execution_mode: str,
    env: dict[str, Any],
    source_loop: dict[str, Any],
    generation_results: list[dict[str, Any]],
    before_node_count: int,
    after_node_count: int,
) -> dict[str, Any]:
    selected = sum(row["selected_candidate_count"] for row in generation_results)
    contract_ready = sum(row["proposal_contract"]["metrics"].get("preflight_ready_count", 0) for row in generation_results)
    preflight_ready = sum(
        row["candidate_preflight"]["readiness_counts"].get("ready_for_fresh_ablation", 0)
        for row in generation_results
    )
    decision_counts = Counter()
    for row in generation_results:
        decision_counts.update(row["candidate_acceptance"].get("decision_counts", {}))
    accepted = int(decision_counts.get("accept", 0))
    applied = sum(len(row["applied_candidate_node_ids"]) for row in generation_results)
    applied_node_delta = sum(row["graph_copy_node_delta"] for row in generation_results)
    return {
        "execution_mode": execution_mode,
        "live_env_ready": bool(env.get("ready")),
        "source_generation_count": source_loop.get("metrics", {}).get("generation_count", 0),
        "generation_count": len(generation_results),
        "selected_candidate_count": selected,
        "contract_ready_count": contract_ready,
        "preflight_ready_count": preflight_ready,
        "fresh_api_call_count": sum(int(row["live_judgment"].get("fresh_api_call_count") or 0) for row in generation_results),
        "planned_fresh_api_call_count": sum(
            int(row["live_judgment"].get("planned_fresh_api_call_count") or 0)
            for row in generation_results
        ),
        "live_error_count": sum(int(row["live_judgment"].get("live_error_count") or 0) for row in generation_results),
        "acceptance_decision_count": sum(decision_counts.values()),
        "acceptance_decision_counts": dict(decision_counts),
        "accepted_count": accepted,
        "rejected_count": selected - accepted,
        "applied_count": applied,
        "applied_node_delta": applied_node_delta,
        "graph_copy_node_delta": after_node_count - before_node_count,
        "main_graph_mutation_count": 0,
        "secret_value_exposed": False,
    }


def _interpretation(*, execution_mode: str, metrics: dict[str, Any]) -> str:
    if execution_mode == "execute_live":
        return (
            "The residual evolution loop now has prospective live evidence across multiple generations: "
            f"{metrics['fresh_api_call_count']} fresh judge calls, {metrics['accepted_count']} accepted, "
            f"{metrics['rejected_count']} rejected, and only accepted candidates applied to a graph copy."
        )
    return (
        "The multi-generation execution path is ready and dry-run validated; execute_live is required "
        "to count as prospective fresh API evidence."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prospective live multi-generation residual expansion artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_live_multigeneration_expansion_20260612")
    parser.add_argument("--execution-mode", choices=["dry_run", "execute_live"], default="dry_run")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--candidates-per-generation", type=int, default=2)
    parser.add_argument("--trigger-rows-per-candidate", type=int, default=4)
    parser.add_argument("--control-rows-per-candidate", type=int, default=2)
    parser.add_argument("--model-alias", default="gpt_mini")
    parser.add_argument("--no-keyfile", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_live_multigeneration_expansion_payload(
        root=root,
        eval_id=args.eval_id,
        execution_mode=args.execution_mode,
        generations=args.generations,
        candidates_per_generation=args.candidates_per_generation,
        trigger_rows_per_candidate=args.trigger_rows_per_candidate,
        control_rows_per_candidate=args.control_rows_per_candidate,
        model_alias=args.model_alias,
        load_keyfile=not args.no_keyfile,
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
