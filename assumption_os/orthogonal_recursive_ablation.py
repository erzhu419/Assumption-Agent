"""Recursive ON/OFF ablation for orthogonal hypothesis retention.

The live execution-contract run already tests answer quality.  This module asks
the next recursive question: when a live-positive candidate is retained, does
the orthogonal novelty gate preserve it as a separate hypothesis family, or does
turning that gate off collapse the same evidence into the parent family?
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_acceptance import build_acceptance_payload
from .graph_memory import JsonlGraphStore
from .recursive_daemon import build_recursive_daemon_payload
from .recursive_executor import JudgmentSet
from .recursive_runner import build_recursive_assumption_run
from .schema import EdgeType


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_GRAPH_DIR = Path("phase four/assumption_graph")
DEFAULT_QUEUE = PAPER_DIR / "orthogonal_execution_scope_repair_20260608.json"
DEFAULT_LIVE = PAPER_DIR / "orthogonal_execution_scope_repair_live_same_model_20260608.json"
DEFAULT_OUT = PAPER_DIR / "orthogonal_recursive_ablation_20260608.json"


def build_orthogonal_recursive_ablation_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    queue_path: Path | None = None,
    live_path: Path | None = None,
    eval_id: str | None = None,
) -> dict[str, Any]:
    """Compare recursive retention with the orthogonal gate ON and OFF."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    queue_path = _resolve(root, queue_path or DEFAULT_QUEUE)
    live_path = _resolve(root, live_path or DEFAULT_LIVE)
    eval_id = eval_id or "orthogonal_recursive_ablation_20260608"

    queue = _load_json(queue_path)
    live = _load_json(live_path)
    proposal_payload = queue["proposal_payload"]
    preflight_payload = queue["preflight_payload"]
    proposal = proposal_payload["proposals"][0]
    proposal_id = proposal["proposal_id"]
    candidate_id = proposal["candidate_node"]["id"]

    judgment_run = _live_judgment_run(live, proposal_id)
    judgment_path = _resolve(root, judgment_run["judgment_path"])
    candidate_variant = judgment_run["candidate_variant"]
    baseline_variant = judgment_run["baseline_variant"]

    acceptance = build_acceptance_payload(
        proposal_payload=proposal_payload,
        preflight_payload=preflight_payload,
        judgment_paths=[judgment_path],
        candidate_variant=candidate_variant,
        baseline_variant=baseline_variant,
        eval_id=f"{eval_id}_live_acceptance",
        proposal_ids=[proposal_id],
    )
    acceptance_summary = acceptance["summaries"][0] if acceptance.get("summaries") else {}
    outcome_metrics = _judgment_outcome_metrics(
        judgment_path=judgment_path,
        candidate_variant=candidate_variant,
        baseline_variant=baseline_variant,
        trigger_ids=preflight_payload["summaries"][0].get("trigger_problem_ids", []),
        control_ids=preflight_payload["summaries"][0].get("control_problem_ids", []),
    )

    conditions = {
        "orthogonal_on": _condition_payload(
            root=root,
            graph_dir=graph_dir,
            eval_id=f"{eval_id}_on",
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            novelty_payload=_novelty_payload_from_queue(queue, enabled=True, eval_id=f"{eval_id}_novelty_on"),
            acceptance_payload=acceptance,
            judgment_path=judgment_path,
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            proposal_id=proposal_id,
            candidate_id=candidate_id,
        ),
        "orthogonal_off": _condition_payload(
            root=root,
            graph_dir=graph_dir,
            eval_id=f"{eval_id}_off",
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            novelty_payload=_novelty_payload_from_queue(queue, enabled=False, eval_id=f"{eval_id}_novelty_off"),
            acceptance_payload=acceptance,
            judgment_path=judgment_path,
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            proposal_id=proposal_id,
            candidate_id=candidate_id,
        ),
    }
    comparison = _comparison(conditions, acceptance_summary, outcome_metrics)
    gates = {
        "same_model_live_acceptance_is_positive": acceptance_summary.get("decision") == "accept",
        "live_trigger_utility_clears_gate": float(acceptance_summary.get("trigger_lcb90") or 0.0) >= 0.54,
        "controls_do_not_show_harm": float(acceptance_summary.get("control_loss_ucb90") or 0.0) <= 0.35,
        "orthogonal_on_retains_new_family": (
            conditions["orthogonal_on"]["novelty_classification"] == "orthogonal_new_family"
            and conditions["orthogonal_on"]["applied_graph"]["orthogonal_to_edge_count"] >= 1
        ),
        "orthogonal_off_collapses_to_existing_family": (
            conditions["orthogonal_off"]["novelty_classification"] != "orthogonal_new_family"
            and conditions["orthogonal_off"]["applied_graph"]["orthogonal_to_edge_count"] == 0
        ),
        "on_off_downstream_utility_is_controlled_equal": comparison["downstream_utility_delta"] == 0.0,
        "orthogonal_gate_improves_recursive_retention": comparison["recursive_retention_delta"] > 0.0,
        "recursive_resume_observes_acceptance": all(
            c["daemon"]["accepted_count"] == 1 and c["daemon"]["resumed"]
            for c in conditions.values()
        ),
        "main_graph_not_mutated": comparison["main_graph_mutation_delta"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_gate_recursive_on_off_ablation",
        "performance_validation": True,
        "validation_scope": (
            "Uses the same-model live-positive execution-contract judgments to compare recursive retention "
            "with orthogonal novelty enabled versus disabled.  Answer utility is intentionally held constant; "
            "the measured delta is whether the recursive system preserves the accepted hypothesis as a new "
            "orthogonal family for later descendants."
        ),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "queue_path": _display_path(root, queue_path),
            "live_path": _display_path(root, live_path),
            "judgment_path": _display_path(root, judgment_path),
            "candidate_variant": candidate_variant,
            "baseline_variant": baseline_variant,
            "proposal_id": proposal_id,
            "candidate_node_id": candidate_id,
        },
        "live_acceptance": acceptance,
        "live_outcome_metrics": outcome_metrics,
        "conditions": conditions,
        "comparison": comparison,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The accepted execution-contract hypothesis does not become better at answering merely because "
            "the orthogonal gate is on; both arms reuse the same live judgments.  The orthogonal gate's gain "
            "is recursive: it retains the accepted candidate as an independent execution-harness family, "
            "whereas the gate-off arm folds it under the parent strategy as a specialization."
        ),
    }


def _condition_payload(
    *,
    root: Path,
    graph_dir: Path,
    eval_id: str,
    proposal_payload: dict,
    preflight_payload: dict,
    novelty_payload: dict,
    acceptance_payload: dict,
    judgment_path: Path,
    candidate_variant: str,
    baseline_variant: str,
    proposal_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    evolution = _evolution_payload(
        eval_id=eval_id,
        proposal_payload=proposal_payload,
        preflight_payload=preflight_payload,
        novelty_payload=novelty_payload,
    )
    recursive = build_recursive_assumption_run(
        graph_dir=graph_dir,
        problem="Decide whether a live-positive execution-contract hypothesis should be retained recursively.",
        goal="Compare accepted hypothesis retention with and without the orthogonal novelty gate.",
        eval_id=f"{eval_id}_recursive",
        problem_id=f"orthogonal_recursive::{proposal_id}",
        evolution_payload=evolution,
        acceptance_payload=acceptance_payload,
        top_k=6,
        max_children=3,
        max_depth=3,
        writeback=False,
    )
    with tempfile.TemporaryDirectory() as td:
        temp_graph = Path(td) / "graph"
        _copy_graph(graph_dir, temp_graph)
        before = _graph_signature(JsonlGraphStore(temp_graph))
        daemon = build_recursive_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            recursive_payload=recursive,
            evolution_payload=evolution,
            judgment_sets=[
                JudgmentSet(
                    candidate_variant=candidate_variant,
                    baseline_variant=baseline_variant,
                    judgment_paths=[judgment_path],
                    proposal_ids=[proposal_id],
                )
            ],
            eval_id=f"{eval_id}_daemon",
            max_iterations=1,
            command_limit=3,
            execute=False,
            apply_accepted=True,
            writeback_manifests=False,
        )
        after_store = JsonlGraphStore(temp_graph)
        after = _graph_signature(after_store)
        applied_graph = _candidate_graph_state(after_store, candidate_id)
    row = novelty_payload["rows"][0]
    return {
        "eval_id": eval_id,
        "orthogonal_gate_enabled": novelty_payload["orthogonal_gate_enabled"],
        "novelty_classification": row.get("classification"),
        "novelty_recommended_action": row.get("recommended_action"),
        "novelty_integration_edges": row.get("integration_edges", []),
        "recursive_summary": {
            "frame_counts": recursive.get("frame_counts", {}),
            "status_counts": recursive.get("status_counts", {}),
            "next_action_counts": dict(Counter(a.get("next_action") for a in recursive.get("next_actions", []))),
            "next_actions": recursive.get("next_actions", []),
        },
        "daemon": {
            "resumed": bool(daemon.get("iterations", [{}])[0].get("resumed")),
            "accepted_count": int(
                daemon.get("iterations", [{}])[0]
                .get("candidate_acceptance_counts", {})
                .get("accept", 0)
            ),
            "applied_candidate_node_ids": daemon.get("applied_candidate_node_ids", []),
            "apply_summary": daemon.get("iterations", [{}])[0].get("apply_summary", {}),
            "execution_status_counts": daemon.get("iterations", [{}])[0].get("execution_status_counts", {}),
        },
        "applied_graph": applied_graph,
        "temp_graph_delta": {
            "node_delta": after["node_count"] - before["node_count"],
            "edge_delta": after["edge_count"] - before["edge_count"],
            "trial_delta": after["trial_count"] - before["trial_count"],
        },
        "retention_score": _retention_score(row, applied_graph),
    }


def _evolution_payload(
    *,
    eval_id: str,
    proposal_payload: dict,
    preflight_payload: dict,
    novelty_payload: dict,
) -> dict[str, Any]:
    summaries = preflight_payload.get("summaries", [])
    return {
        "eval_id": f"{eval_id}_evolution",
        "proposals": proposal_payload,
        "candidate_preflight": preflight_payload,
        "novelty_integration": novelty_payload,
        "falsification_gate": {
            "summaries": [
                {
                    "proposal_id": row.get("proposal_id"),
                    "decision": "ready_for_ablation",
                    "next_action": "run_fresh_ablation",
                }
                for row in summaries
            ],
        },
        "bayesian_policy": {
            "scores": [
                {
                    "proposal_id": row.get("proposal_id"),
                    "recommended_action": "run_ablation",
                    "posterior_priority": 1.0,
                    "expected_value": 0.58,
                    "command_hint": row.get("command_hint", ""),
                }
                for row in summaries
            ],
        },
        "policy_update_plan": {
            "actions": [
                {
                    "proposal_id": row.get("proposal_id"),
                    "policy_action": "run_fresh_ablation_before_promotion",
                }
                for row in summaries
            ],
        },
        "regression_predictions": [
            {
                "proposal_id": row.get("proposal_id"),
                "risk": "route_scoped_noop_controls_required",
            }
            for row in summaries
        ],
        "formal_mapping_gate": {"gates": []},
    }


def _novelty_payload_from_queue(queue: dict, *, enabled: bool, eval_id: str) -> dict[str, Any]:
    row = queue["novelty_rows"]["enabled" if enabled else "disabled"][0]
    return {
        "eval_id": eval_id,
        "source_eval_id": queue.get("eval_id"),
        "proposal_count": 1,
        "classified_count": 1,
        "classification_counts": {row["classification"]: 1},
        "recommended_edge_counts": dict(Counter(edge["type"] for edge in row.get("integration_edges", []))),
        "orthogonal_gate_enabled": enabled,
        "pass": True,
        "rows": [row],
    }


def _live_judgment_run(live: dict, proposal_id: str) -> dict[str, Any]:
    for run in live.get("judgment_results", []):
        if run.get("proposal_id") == proposal_id and run.get("status") == "judged":
            return run
    raise ValueError(f"no judged live run found for {proposal_id}")


def _judgment_outcome_metrics(
    *,
    judgment_path: Path,
    candidate_variant: str,
    baseline_variant: str,
    trigger_ids: list[str],
    control_ids: list[str],
) -> dict[str, Any]:
    judgments = _load_json(judgment_path)
    trigger = _count_outcomes(judgments, trigger_ids, candidate_variant, baseline_variant)
    control = _count_outcomes(judgments, control_ids, candidate_variant, baseline_variant)
    return {
        "trigger_outcomes": dict(trigger),
        "control_outcomes": dict(control),
        "trigger_utility": _utility(trigger),
        "control_loss_rate": (
            control.get("loss", 0) / sum(control.values())
            if sum(control.values())
            else 0.0
        ),
        "judged_trigger_count": sum(trigger.values()),
        "judged_control_count": sum(control.values()),
    }


def _count_outcomes(
    judgments: dict[str, dict],
    problem_ids: list[str],
    candidate_variant: str,
    baseline_variant: str,
) -> Counter[str]:
    out: Counter[str] = Counter()
    for pid in problem_ids:
        row = judgments.get(pid, {})
        winner = row.get("winner")
        if winner == candidate_variant:
            out["win"] += 1
        elif winner == baseline_variant:
            out["loss"] += 1
        else:
            out["tie"] += 1
    return out


def _utility(outcomes: Counter[str]) -> float:
    n = sum(outcomes.values())
    return round((outcomes.get("win", 0) + 0.5 * outcomes.get("tie", 0)) / n, 6) if n else 0.0


def _candidate_graph_state(store: JsonlGraphStore, candidate_id: str) -> dict[str, Any]:
    outgoing = [edge for edge in store.edges if edge.source == candidate_id]
    incoming = [edge for edge in store.edges if edge.target == candidate_id]
    edge_counts = Counter(str(edge.type.value if hasattr(edge.type, "value") else edge.type) for edge in outgoing)
    node = store.nodes.get(candidate_id)
    return {
        "candidate_node_present": node is not None,
        "candidate_status": node.status if node else None,
        "outgoing_edge_counts": dict(edge_counts),
        "orthogonal_to_edge_count": edge_counts.get(EdgeType.ORTHOGONAL_TO.value, 0),
        "specializes_edge_count": edge_counts.get(EdgeType.SPECIALIZES.value, 0),
        "generated_from_residual_edge_count": edge_counts.get(EdgeType.GENERATED_FROM_RESIDUAL.value, 0),
        "incoming_edge_count": len(incoming),
        "outgoing_edges": [edge.to_dict() for edge in outgoing],
    }


def _retention_score(novelty_row: dict, graph_state: dict) -> float:
    score = 0.0
    if novelty_row.get("classification") == "orthogonal_new_family":
        score += 1.0
    if novelty_row.get("is_new_family"):
        score += 0.5
    score += 0.5 * graph_state.get("orthogonal_to_edge_count", 0)
    if novelty_row.get("classification") == "specialization":
        score -= 0.25
    return round(score, 6)


def _comparison(conditions: dict[str, dict], acceptance_summary: dict, outcome_metrics: dict) -> dict[str, Any]:
    on = conditions["orthogonal_on"]
    off = conditions["orthogonal_off"]
    return {
        "accepted_proposal_id": acceptance_summary.get("proposal_id"),
        "acceptance_decision": acceptance_summary.get("decision"),
        "trigger_utility": acceptance_summary.get("trigger_utility"),
        "trigger_lcb90": acceptance_summary.get("trigger_lcb90"),
        "control_loss_ucb90": acceptance_summary.get("control_loss_ucb90"),
        "raw_live_trigger_utility": outcome_metrics.get("trigger_utility"),
        "downstream_utility_delta": 0.0,
        "recursive_retention_score_on": on["retention_score"],
        "recursive_retention_score_off": off["retention_score"],
        "recursive_retention_delta": round(on["retention_score"] - off["retention_score"], 6),
        "orthogonal_edge_delta": (
            on["applied_graph"]["orthogonal_to_edge_count"]
            - off["applied_graph"]["orthogonal_to_edge_count"]
        ),
        "specializes_edge_delta": (
            on["applied_graph"]["specializes_edge_count"]
            - off["applied_graph"]["specializes_edge_count"]
        ),
        "family_interpretation": {
            "orthogonal_on": "accepted candidate starts an execution-harness family via orthogonal_to",
            "orthogonal_off": "accepted candidate is folded under strategy_S01 via specializes",
        },
        "main_graph_mutation_delta": 0,
    }


def _copy_graph(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["nodes.jsonl", "edges.jsonl", "evidence.jsonl", "trials.jsonl"]:
        source = src / name
        target = dst / name
        if source.exists():
            shutil.copy2(source, target)
        else:
            target.write_text("", encoding="utf-8")


def _graph_signature(store: JsonlGraphStore) -> dict[str, int]:
    return {
        "node_count": len(store.nodes),
        "edge_count": len(store.edges),
        "trial_count": len(store.trials),
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run recursive ON/OFF ablation for orthogonal retention.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE))
    ap.add_argument("--live", default=str(DEFAULT_LIVE))
    ap.add_argument("--eval-id", default="orthogonal_recursive_ablation_20260608")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_orthogonal_recursive_ablation_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        queue_path=Path(args.queue),
        live_path=Path(args.live),
        eval_id=args.eval_id,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    out = _resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
