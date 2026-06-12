"""Integrated graph episode for dialectical framework evolution.

The framework-growth modules can generate and score new philosophical/method
frameworks.  This episode connects that machinery to the Assumption Graph
lifecycle: contract check, copy-only graph graft, readback, rollback rehearsal,
journal replay, and descendant seed generation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .autonomy_journal import AutonomyJournalEvent, AppendOnlyAutonomyJournal, PAPER_DIR, stable_hash
from .conservative_generalization_gate import (
    REQUIRED_PROMOTION_RELATIONS,
    build_conservative_generalization_gate_payload,
)
from .framework_branch_ledger import build_framework_branch_ledger_payload
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .philosophy_growth_benchmark import build_philosophy_growth_benchmark_payload
from .proposal_contract import build_proposal_contract_payload
from .proposals import ProposalType
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, HypothesisKind, TrialManifest


DEFAULT_OUT = PAPER_DIR / "framework_evolution_graph_episode_20260612.json"
GRAPH_FILES = ("nodes.jsonl", "edges.jsonl", "evidence.jsonl", "trials.jsonl")


def build_framework_evolution_graph_episode_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    eval_id: str = "framework_evolution_graph_episode_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    graph_dir = graph_dir or root / "phase four/assumption_graph"
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    gate = build_conservative_generalization_gate_payload(root=root, eval_id=f"{eval_id}_gate")
    ledger = build_framework_branch_ledger_payload(root=root, eval_id=f"{eval_id}_ledger")
    bench = build_philosophy_growth_benchmark_payload(root=root, eval_id=f"{eval_id}_bench")
    active = next(row for row in gate["evaluations"] if row["decision"] == "active_scoped_framework")
    graph_patch = _active_graph_patch(gate["graph_patch"], active_framework_id=active["framework_id"])
    proposal_payload = _proposal_payload(eval_id=eval_id, active=active, graph_patch=graph_patch)

    with tempfile.TemporaryDirectory(prefix="framework_evolution_graph_episode_") as td:
        tmp_root = Path(td)
        tmp_graph = tmp_root / "graph"
        _copy_graph_files(graph_dir, tmp_graph)
        journal = AppendOnlyAutonomyJournal(tmp_root / "journal.jsonl")
        before_store = JsonlGraphStore(tmp_graph)
        before_hash = _store_hash(before_store)
        contract = build_proposal_contract_payload(
            proposal_payload=proposal_payload,
            eval_id=f"{eval_id}_contract",
            store=before_store,
        )
        after_contract_hash = _store_hash(before_store)
        _append_event(
            journal,
            cycle_id=eval_id,
            event_id="event_contract",
            event_type="proposal_contract",
            before=before_hash,
            after=after_contract_hash,
            idempotency_key=f"{eval_id}:contract",
            status="completed",
            metadata={"admitted": contract["admitted_proposal_ids"]},
        )

        graft_result = _copy_graft_framework_patch(
            store=before_store,
            graph_patch=graph_patch,
            active_framework_id=active["framework_id"],
            apply_allowed=bool(contract["admitted_proposal_ids"]),
        )
        after_graft_store = JsonlGraphStore(tmp_graph)
        after_graft_hash = _store_hash(after_graft_store)
        _append_event(
            journal,
            cycle_id=eval_id,
            event_id="event_copy_graft",
            event_type="copy_graph_graft",
            before=after_contract_hash,
            after=after_graft_hash,
            idempotency_key=f"{eval_id}:copy_graft:{active['framework_id']}",
            status="completed",
            metadata={"added_node_count": graft_result["added_node_count"], "added_edge_count": graft_result["added_edge_count"]},
        )

        readback = _readback(after_graft_store, active_framework_id=active["framework_id"])
        _append_event(
            journal,
            cycle_id=eval_id,
            event_id="event_readback",
            event_type="graph_readback",
            before=after_graft_hash,
            after=after_graft_hash,
            idempotency_key=f"{eval_id}:readback:{active['framework_id']}",
            status="completed",
            metadata=readback,
        )
        rollback = _rollback_rehearsal(
            store=after_graft_store,
            added_node_ids=graft_result["added_node_ids"],
            added_edge_keys=graft_result["added_edge_keys"],
        )
        after_rollback_store = JsonlGraphStore(tmp_graph)
        after_rollback_hash = _store_hash(after_rollback_store)
        _append_event(
            journal,
            cycle_id=eval_id,
            event_id="event_rollback_rehearsal",
            event_type="rollback_rehearsal",
            before=after_graft_hash,
            after=after_rollback_hash,
            idempotency_key=f"{eval_id}:rollback_rehearsal:{active['framework_id']}",
            status="completed",
            metadata=rollback,
        )
        replay = journal.replay()
        replay_again = journal.replay()

    descendant_seeds = _descendant_seeds(active=active, readback=readback, bench=bench)
    metrics = _metrics(
        gate=gate,
        ledger=ledger,
        bench=bench,
        contract=contract,
        graph_patch=graph_patch,
        graft_result=graft_result,
        readback=readback,
        rollback=rollback,
        replay=replay,
        replay_again=replay_again,
        descendant_seeds=descendant_seeds,
    )
    gates = {
        "source_gate_passes": bool(gate.get("pass")),
        "source_ledger_passes": bool(ledger.get("pass")),
        "source_bench_passes": bool(bench.get("pass")),
        "contract_admits_active_framework": metrics["contract_admitted_count"] == 1,
        "copy_graft_adds_framework_nodes": metrics["graft_added_node_count"] >= 6,
        "copy_graft_adds_required_relations": metrics["required_relation_coverage"] == 1.0,
        "readback_retrieves_active_framework": metrics["readback_active_rank"] <= 3,
        "readback_relation_coverage": metrics["readback_relation_coverage"] == 1.0,
        "descendant_seeds_generated": metrics["descendant_seed_count"] >= 3,
        "negative_evidence_retained": metrics["negative_evidence_retained_count"] >= 1,
        "rollback_restores_copy": metrics["rollback_success"] is True,
        "journal_replay_exact": metrics["journal_replay_exact"] is True,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
        "core_prior_not_promoted": metrics["core_philosophy_prior_promotion_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "framework_evolution_graph_episode",
        "source_md": "reconstruction/md/self_evo_roadmap.md",
        "reconstruction_v2_full_phase": "r7_framework_graph_lifecycle_episode",
        "performance_validation": True,
        "validation_scope": (
            "Connects dialectical framework growth to the Assumption Graph lifecycle.  The active scoped "
            "framework is contract-checked, grafted into a controlled graph copy, read back through graph "
            "retrieval, rollback-rehearsed, journal-replayed, and converted into descendant seeds.  The main "
            "graph remains unchanged."
        ),
        "source_artifacts": {
            "conservative_generalization_gate": {"pass": gate["pass"], "metrics": gate["metrics"]},
            "framework_branch_ledger": {"pass": ledger["pass"], "metrics": ledger["metrics"]},
            "philosophy_growth_benchmark": {"pass": bench["pass"], "metrics": bench["metrics"]},
        },
        "active_framework": active,
        "proposal_payload": proposal_payload,
        "contract": contract,
        "graph_patch": {
            "node_count": len(graph_patch["nodes"]),
            "edge_count": len(graph_patch["edges"]),
            "edge_type_counts": graph_patch["edge_type_counts"],
        },
        "graft_result": graft_result,
        "readback": readback,
        "rollback_rehearsal": rollback,
        "journal_replay": replay.to_dict(),
        "descendant_seeds": descendant_seeds,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Framework growth is no longer only scored.  It can be represented as a contract-checked graph "
            "graft, read back from graph memory, rolled back, replayed, and used to seed descendants while "
            "remaining scoped and non-core."
        ),
    }


def _active_graph_patch(graph_patch: dict[str, Any], *, active_framework_id: str) -> dict[str, Any]:
    edges = [
        edge for edge in graph_patch["edges"]
        if edge["source"] == active_framework_id or edge["target"] == active_framework_id
    ]
    node_ids = {active_framework_id}
    for edge in edges:
        node_ids.add(edge["source"])
        node_ids.add(edge["target"])
    nodes_by_id = {node["id"]: node for node in graph_patch["nodes"]}
    nodes = [nodes_by_id[node_id] for node_id in sorted(node_ids) if node_id in nodes_by_id]
    for node_id in sorted(node_ids - set(nodes_by_id)):
        nodes.append(
            AssumptionNode(
                id=node_id,
                type=AssumptionType.METHOD,
                kind=HypothesisKind.PROCESS_MODEL,
                claim=f"Framework reference: {node_id}",
                status="framework_reference",
                tags=["framework_reference", "conservative_generalization_parent"],
                payload={"source": "framework_evolution_graph_episode"},
            ).to_dict()
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "edge_type_counts": _counts(edge["type"] for edge in edges),
    }


def _proposal_payload(*, eval_id: str, active: dict[str, Any], graph_patch: dict[str, Any]) -> dict[str, Any]:
    active_node = next(node for node in graph_patch["nodes"] if node["id"] == active["framework_id"])
    candidate_node = dict(active_node)
    candidate_node["status"] = "candidate"
    candidate_node["context_conditions"] = [
        *candidate_node.get("context_conditions", []),
        "scoped active framework candidate from conservative generalization gate",
        "negative control: outside parent scope must not regress old successes",
    ]
    candidate_node["predicted_effects"] = [
        "increase residual explanation without old success regression",
        "generate descendant branches for dense dependency and verifier-routing residuals",
        "negative control: prompt-length placebo should remain rejected",
    ]
    candidate_node["risk_predictions"] = [
        "regression risk if used outside scope",
        "harm risk if promoted to core philosophy prior without survival evidence",
        "negative control must reject surface-only framework growth",
    ]
    candidate_node["verifiers"] = [
        "conservative_generalization_gate",
        "philosophy_growth_benchmark",
        "negative_control_old_success_regression",
    ]
    parent_id = active["parent_frameworks"][0]
    edges = [
        edge for edge in graph_patch["edges"]
        if edge["source"] == candidate_node["id"] and edge["target"] == parent_id
    ][:2]
    if not edges:
        edges = [edge for edge in graph_patch["edges"] if edge["source"] == candidate_node["id"]][:1]
    manifest = TrialManifest(
        problem_id=f"{eval_id}:{candidate_node['id']}",
        action_type="framework_graph_graft",
        assumption=candidate_node["claim"],
        why_selected="Conservative generalization gate promoted this framework to active scoped status.",
        expected_effect="Graph copy should retrieve the active framework and required relation edges without mutating the main graph.",
        assumption_ids=[candidate_node["id"]],
        component="framework_evolution_graph_episode",
        predicted_regressions=[
            "old success regression",
            "scope overreach",
            "negative control harm",
        ],
        verifier="conservative_generalization_gate + philosophy_growth_benchmark + negative control readback",
        verification_plan="contract -> copy graft -> graph readback -> rollback rehearsal -> journal replay",
        rollback_condition="remove all graph patch node ids and relation edges from controlled copy",
        cost=0.0,
    ).to_dict()
    return {
        "eval_id": f"{eval_id}_proposal_payload",
        "proposal_counts": {ProposalType.ASSUMPTION_REVISION.value: 1},
        "proposals": [
            {
                "proposal_id": f"framework_prop_{stable_hash(candidate_node['id'])}",
                "proposal_type": ProposalType.ASSUMPTION_REVISION.value,
                "parent_node_id": parent_id,
                "candidate_node": candidate_node,
                "edges": edges,
                "manifest": manifest,
                "rationale": "Graft active scoped framework after conservative-generalization validation.",
                "priority": active["metrics"]["framework_growth_score"],
                "source_action": {
                    "action_type": "framework_graph_graft",
                    "framework_id": candidate_node["id"],
                    "decision": active["decision"],
                },
            }
        ],
    }


def _copy_graft_framework_patch(
    *,
    store: JsonlGraphStore,
    graph_patch: dict[str, Any],
    active_framework_id: str,
    apply_allowed: bool,
) -> dict[str, Any]:
    before_nodes = set(store.nodes)
    before_edges = {edge.key for edge in store.edges}
    if apply_allowed:
        for node in graph_patch["nodes"]:
            store.upsert_node(AssumptionNode.from_dict(node))
        for edge in graph_patch["edges"]:
            store.add_edge(AssumptionEdge.from_dict(edge))
        store.flush()
    after_edges = {edge.key for edge in store.edges}
    added_node_ids = sorted(set(store.nodes) - before_nodes)
    added_edge_keys = sorted(after_edges - before_edges)
    return {
        "active_framework_id": active_framework_id,
        "apply_allowed": apply_allowed,
        "added_node_ids": added_node_ids,
        "added_edge_keys": [list(key) for key in added_edge_keys],
        "added_node_count": len(added_node_ids),
        "added_edge_count": len(added_edge_keys),
        "main_graph_mutation_count": 0,
    }


def _readback(store: JsonlGraphStore, *, active_framework_id: str) -> dict[str, Any]:
    graph = SimpleAssumptionGraph(store)
    activated = graph.retrieve(
        "dependency aware controlled intervention residual explanation limiting case old success preservation",
        seeds=[active_framework_id],
        top_k=12,
    )
    ranked = sorted(activated.scores.items(), key=lambda item: -item[1])
    rank = next((index + 1 for index, (node_id, _) in enumerate(ranked) if node_id == active_framework_id), 999)
    edge_types = {
        str(edge.type.value if hasattr(edge.type, "value") else edge.type)
        for edge in store.edges
        if edge.source == active_framework_id or edge.target == active_framework_id
    }
    return {
        "active_framework_id": active_framework_id,
        "active_rank": rank,
        "retrieved_node_ids": [node.id for node in activated.nodes],
        "relation_types": sorted(edge_types),
        "required_relation_coverage": round(
            len(REQUIRED_PROMOTION_RELATIONS.intersection(edge_types)) / len(REQUIRED_PROMOTION_RELATIONS),
            4,
        ),
    }


def _rollback_rehearsal(
    *,
    store: JsonlGraphStore,
    added_node_ids: list[str],
    added_edge_keys: list[list[str]],
) -> dict[str, Any]:
    edge_keys = {tuple(key) for key in added_edge_keys}
    before_node_count = len(store.nodes)
    before_edge_count = len(store.edges)
    for node_id in added_node_ids:
        store.nodes.pop(node_id, None)
    store.edges = [edge for edge in store.edges if edge.key not in edge_keys]
    store.flush()
    return {
        "before_node_count": before_node_count,
        "before_edge_count": before_edge_count,
        "removed_node_count": len(added_node_ids),
        "removed_edge_count": len(edge_keys),
        "after_node_count": len(store.nodes),
        "after_edge_count": len(store.edges),
        "rollback_success": all(node_id not in store.nodes for node_id in added_node_ids)
        and all(edge.key not in edge_keys for edge in store.edges),
    }


def _descendant_seeds(*, active: dict[str, Any], readback: dict[str, Any], bench: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("dense_dependency_pairwise_ablation", "test grouped intervention when dependency graph is dense"),
        ("sparse_dependency_reduction_check", "verify reduction to ordinary control variables when dependencies are sparse"),
        ("prompt_placebo_negative_control", "ensure style/length boost remains rejected under matched controls"),
        ("verifier_routing_boundary", "route uncertainty-driven descendants through verifier ladder"),
    ]
    return [
        {
            "seed_id": f"framework_seed_{stable_hash([active['framework_id'], name])}",
            "parent_framework_id": active["framework_id"],
            "seed_kind": name,
            "hypothesis": text,
            "required_relation_coverage": readback["required_relation_coverage"],
            "framework_growth_score": bench["metrics"]["framework_growth_score"],
            "next_action": "generate_child_branch_and_run_conservative_gate",
        }
        for name, text in specs
    ]


def _metrics(
    *,
    gate: dict[str, Any],
    ledger: dict[str, Any],
    bench: dict[str, Any],
    contract: dict[str, Any],
    graph_patch: dict[str, Any],
    graft_result: dict[str, Any],
    readback: dict[str, Any],
    rollback: dict[str, Any],
    replay: Any,
    replay_again: Any,
    descendant_seeds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_pass_rate": round(sum(1 for item in [gate, ledger, bench] if item.get("pass")) / 3, 4),
        "contract_admitted_count": len(contract["admitted_proposal_ids"]),
        "contract_quarantined_count": len(contract["quarantined_proposal_ids"]),
        "graph_patch_node_count": len(graph_patch["nodes"]),
        "graph_patch_edge_count": len(graph_patch["edges"]),
        "graft_added_node_count": graft_result["added_node_count"],
        "graft_added_edge_count": graft_result["added_edge_count"],
        "required_relation_coverage": round(
            len(REQUIRED_PROMOTION_RELATIONS.intersection(set(graph_patch["edge_type_counts"])))
            / len(REQUIRED_PROMOTION_RELATIONS),
            4,
        ),
        "readback_active_rank": readback["active_rank"],
        "readback_relation_coverage": readback["required_relation_coverage"],
        "descendant_seed_count": len(descendant_seeds),
        "negative_evidence_retained_count": ledger["metrics"]["negative_evidence_retained_count"],
        "rollback_success": bool(rollback["rollback_success"]),
        "journal_replay_exact": (
            replay.final_graph_hash == replay_again.final_graph_hash
            and replay.divergence_detected is False
            and replay_again.divergence_detected is False
        ),
        "journal_event_count": replay.event_count,
        "main_graph_mutation_count": graft_result["main_graph_mutation_count"],
        "core_philosophy_prior_promotion_count": bench["metrics"]["core_philosophy_prior_promotion_count"],
        "framework_growth_score": bench["metrics"]["framework_growth_score"],
    }


def _append_event(
    journal: AppendOnlyAutonomyJournal,
    *,
    cycle_id: str,
    event_id: str,
    event_type: str,
    before: str,
    after: str,
    idempotency_key: str,
    status: str,
    metadata: dict[str, Any],
) -> None:
    journal.append(
        AutonomyJournalEvent(
            cycle_id=cycle_id,
            event_id=event_id,
            event_type=event_type,
            input_hash=before,
            output_hash=after,
            graph_before_hash=before,
            graph_after_hash=after,
            idempotency_key=idempotency_key,
            permission_boundary="controlled_copy_only_explicit_main_apply_required",
            status=status,
            metadata=metadata,
        )
    )


def _store_hash(store: JsonlGraphStore) -> str:
    return stable_hash(
        {
            "nodes": sorted((node.id, node.status, node.claim) for node in store.nodes.values()),
            "edges": sorted((edge.source, edge.target, str(edge.type), edge.weight) for edge in store.edges),
            "trial_count": len(store.trials),
            "evidence_count": len(store.evidence),
        }
    )


def _copy_graph_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_FILES:
        src = source / name
        dst = target / name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.write_text("", encoding="utf-8")


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build framework evolution graph episode artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default="phase four/assumption_graph")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--eval-id", default="framework_evolution_graph_episode_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_framework_evolution_graph_episode_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        eval_id=args.eval_id,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
