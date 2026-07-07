"""Next-generation live queue for the accepted orthogonal descendant.

The previous live run accepted ``cand_f8ca2582dbc4`` but also exposed one
negative trigger row.  This module turns that real residual into a stricter
generation-4 descendant: retain only the live-positive scope, abstain on the
software migration row, and add concrete bridges synthesized from the failed
child's live judgment residuals.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_eval import build_candidate_eval_payload
from .graph_memory import JsonlGraphStore
from .novelty_integration import build_novelty_integration_payload
from .orthogonal_descendant_live_queue import (
    DEFAULT_OUT as DEFAULT_PARENT_QUEUE,
    DEFAULT_RETAINED_GRAPH_DIR as DEFAULT_SOURCE_GRAPH_DIR,
    _commands_are_secret_free,
    _compact_daemon,
    _copy_graph,
    _display_path,
    _env_status,
    _load_json,
    _resolve,
)
from .orthogonal_execution_queue import DEFAULT_META_PATH, DEFAULT_SAMPLE_PATH
from .orthogonal_recursive_ablation import PAPER_DIR
from .recursive_daemon import build_preflight_queue_daemon_payload
from .recursive_executor import JudgmentSet
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    stable_id,
)


DEFAULT_OUT = PAPER_DIR / "orthogonal_descendant_nextgen_live_queue_20260609.json"
DEFAULT_PROPOSALS_OUT = PAPER_DIR / "orthogonal_descendant_nextgen_live_queue_proposals_20260609.json"
DEFAULT_PREFLIGHT_OUT = PAPER_DIR / "orthogonal_descendant_nextgen_live_queue_preflight_20260609.json"
DEFAULT_RETAINED_GRAPH_DIR = PAPER_DIR / "orthogonal_descendant_nextgen_live_graph_20260609"

PARENT_PROPOSAL_ID = "prop_d7abf65010d2"
PARENT_CANDIDATE_ID = "cand_f8ca2582dbc4"
SEED_CANDIDATE_ID = "cand_39de0aeae8a3"

TRIGGER_IDS = [
    "business_0097",
    "business_0192",
    "business_0218",
    "daily_life_0173",
]
ABSTAINED_RESIDUAL_IDS = [
    "software_engineering_0142",
]


def build_orthogonal_descendant_nextgen_live_queue_payload(
    *,
    root: Path,
    source_graph_dir: Path | None = None,
    parent_queue_path: Path | None = None,
    sample_path: Path | None = None,
    meta_path: Path | None = None,
    eval_id: str | None = None,
    proposals_out: Path | None = None,
    preflight_out: Path | None = None,
    retained_graph_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a live-ready queue for the next descendant generation."""

    root = root.resolve()
    source_graph_dir = _resolve(root, source_graph_dir or DEFAULT_SOURCE_GRAPH_DIR)
    parent_queue_path = _resolve(root, parent_queue_path or DEFAULT_PARENT_QUEUE)
    sample_path = _resolve(root, sample_path or DEFAULT_SAMPLE_PATH)
    meta_path = _resolve(root, meta_path or DEFAULT_META_PATH)
    proposals_out = _resolve(root, proposals_out or DEFAULT_PROPOSALS_OUT)
    preflight_out = _resolve(root, preflight_out or DEFAULT_PREFLIGHT_OUT)
    retained_graph_dir = _resolve(root, retained_graph_dir or DEFAULT_RETAINED_GRAPH_DIR)
    eval_id = eval_id or "orthogonal_descendant_nextgen_live_queue_20260609"

    retained_graph = _prepare_nextgen_graph_snapshot(
        source_graph_dir=source_graph_dir,
        retained_graph_dir=retained_graph_dir,
        parent_queue_path=parent_queue_path,
    )
    proposal_payload = _build_proposal_payload(eval_id=eval_id)
    proposal_id = proposal_payload["proposals"][0]["proposal_id"]
    store = JsonlGraphStore(retained_graph_dir)
    novelty = build_novelty_integration_payload(
        store,
        proposal_payload,
        eval_id=f"{eval_id}_novelty",
        enable_orthogonal=True,
    )
    preflight = build_candidate_eval_payload(
        graph_dir=retained_graph_dir,
        proposal_payload=proposal_payload,
        sample=_load_json(sample_path),
        meta_by_pid=_load_json(meta_path),
        eval_id=f"{eval_id}_preflight",
        proposal_ids=[proposal_id],
        min_trigger_n=4,
        min_active_trigger_n=4,
        force_proposal_route=True,
        max_control_ids=8,
        proposals_arg=_display_path(root, proposals_out),
        sample_arg=_display_path(root, sample_path),
        graph_arg=_display_path(root, retained_graph_dir),
    )
    evolution = _evolution_payload(
        eval_id=eval_id,
        proposal_payload=proposal_payload,
        preflight=preflight,
        novelty=novelty,
    )
    readback = _fixture_readback(
        root=root,
        graph_dir=retained_graph_dir,
        proposal_payload=proposal_payload,
        preflight=preflight,
        evolution=evolution,
        novelty=novelty,
        eval_id=eval_id,
    )

    proposals_out.parent.mkdir(parents=True, exist_ok=True)
    preflight_out.parent.mkdir(parents=True, exist_ok=True)
    proposals_out.write_text(json.dumps(proposal_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preflight_out.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = _metrics(
        proposal_payload=proposal_payload,
        novelty=novelty,
        preflight=preflight,
        readback=readback,
        retained_graph=retained_graph,
        env=_env_status(),
    )
    gates = {
        "single_nextgen_descendant": metrics["proposal_count"] == 1,
        "parent_live_descendant_retained": (
            retained_graph["parent_candidate_node_id"] == PARENT_CANDIDATE_ID
            and retained_graph["parent_status_after_snapshot"] == "active"
        ),
        "classified_as_specialization_of_accepted_descendant": (
            metrics["specialization_count"] == 1
            and metrics["specializes_edge_count"] == 1
        ),
        "ready_for_fresh_ablation": metrics["preflight_ready_count"] == 1,
        "retained_live_positive_scope_only": (
            metrics["trigger_count"] == 4
            and metrics["active_trigger_count"] == 4
            and set(metrics["trigger_problem_ids"]) == set(TRIGGER_IDS)
        ),
        "software_negative_residual_abstained": ABSTAINED_RESIDUAL_IDS[0] not in metrics["trigger_problem_ids"],
        "controls_present": metrics["control_count"] >= 8,
        "no_outside_activation": metrics["outside_active_count"] == 0,
        "fixture_readback_accepts": metrics["readback_accept_count"] == 1,
        "readback_without_apply_does_not_mutate_graph": (
            metrics["readback_applied_count"] == 0
            and not metrics["node_mutation_without_apply"]
        ),
        "fixture_temp_apply_writes_candidate": metrics["apply_applied_count"] == 1,
        "commands_are_secret_free": _commands_are_secret_free(_next_commands(root, proposal_id, proposals_out, preflight_out)),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_descendant_nextgen_live_ready_queue",
        "performance_validation": True,
        "validation_scope": (
            "Builds a generation-4 descendant from the real live residual of the accepted generation-3 "
            "constraint bridge.  This is a scope-retention plus residual-specific bridge test: keep the "
            "live-positive trigger rows, abstain on the observed software migration loss, inject the "
            "concrete bridge fields exposed by the failed child judgment, and require the same fresh "
            "answer/judge gate before retention."
        ),
        "status": "live_ready" if metrics["live_env_ready"] else "live_ready_env_missing",
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "source_graph_dir": _display_path(root, source_graph_dir),
            "retained_graph_dir": _display_path(root, retained_graph_dir),
            "parent_queue_path": _display_path(root, parent_queue_path),
            "retained_graph_snapshot": retained_graph,
            "sample_path": _display_path(root, sample_path),
            "meta_path": _display_path(root, meta_path),
            "proposals_out": _display_path(root, proposals_out),
            "preflight_out": _display_path(root, preflight_out),
            "trigger_problem_ids": TRIGGER_IDS,
            "abstained_residual_problem_ids": ABSTAINED_RESIDUAL_IDS,
        },
        "proposal_payload": proposal_payload,
        "novelty_summary": {
            "classification_counts": novelty.get("classification_counts", {}),
            "recommended_edge_counts": novelty.get("recommended_edge_counts", {}),
            "rows": novelty.get("rows", []),
        },
        "preflight_payload": preflight,
        "daemon_validation": readback,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "next_commands": _next_commands(root, proposal_id, proposals_out, preflight_out),
        "interpretation": (
            "This candidate is not a new orthogonal family.  It is a selective-retention descendant of the "
            "accepted constraint bridge: keep the rows where the parent improved answer quality, stop firing "
            "on the observed software migration residual, and use the failed child judgment to add more concrete "
            "city/channel, secrecy-contract, and staged-transition bridge fields."
        ),
    }


def _build_proposal_payload(*, eval_id: str) -> dict[str, Any]:
    candidate_id = stable_id("cand", eval_id, "residual_specific_retained_scope_execution_bridge_v4")
    proposal_id = stable_id("prop", eval_id, candidate_id)
    candidate = AssumptionNode(
        id=candidate_id,
        type=AssumptionType.HARNESS,
        kind=HypothesisKind.VERIFICATION,
        claim=(
            "残差具体化执行桥接后代假设：当上一代约束保持执行桥接已在商业转型、投放归因、"
            "品类上线和通勤试错题上形成正收益时，下一代不能只保留清单，而要把 live judgment 暴露的"
            "缺失桥接补上。投放题必须把城市/渠道先验写成可更新后验，分开转化指标和品牌指标，说明"
            "放量后的边际衰减；产线/外包题必须写明保密分层、合同/交付结构、专家知识交付窗口和"
            "stop-loss 决策点；教育转型题必须给两年分阶段路径、课程/城市分层、名师/教研/销售激励"
            "对齐和回滚门槛；通勤题必须用真实时段 P95/最大值试错而不是只给理论路线。"
            "对上一代输掉的深技术迁移/API 治理题仍先 abstain，等待 technical-migration 子假设修复。"
        ),
        formal_form={
            "category_inspired_diagram": {
                "objects": [
                    "execution_contract_seed",
                    "constraint_preserving_bridge_v2",
                    "retained_positive_scope_v3_failed",
                    "residual_specific_bridge_v4",
                ],
                "morphisms": [
                    {
                        "source": "execution_contract_seed",
                        "target": "constraint_preserving_bridge_v2",
                        "preserves": ["reversible_trial", "metric_gate", "rollback_path"],
                        "adds": ["hard_constraint_preservation", "role_specific_bridge"],
                    },
                    {
                        "source": "constraint_preserving_bridge_v2",
                        "target": "retained_positive_scope_v3_failed",
                        "preserves": ["hard_constraint_preservation", "compact_execution_manifest"],
                        "adds": ["live_residual_abstain_boundary", "selective_retention_scope"],
                    },
                    {
                        "source": "retained_positive_scope_v3_failed",
                        "target": "residual_specific_bridge_v4",
                        "preserves": ["selective_retention_scope", "live_residual_abstain_boundary"],
                        "adds": [
                            "city_channel_posterior_update",
                            "secrecy_layered_contract_bridge",
                            "two_year_segmented_transition_path",
                        ],
                    },
                ],
                "invariants": [
                    "do_not_relax_user_given_constraints",
                    "name_current_baseline_before_intervention",
                    "define_reversible_minimum_trial_and_stop_rule",
                    "bridge_execution_fields_to_problem_specific_risk",
                ],
                "negative_controls": ABSTAINED_RESIDUAL_IDS,
                "certificate_type": "bounded_structural_morphism_proof_lite",
            }
        },
        context_conditions=[
            "问题属于已 live-positive 的商业转型、广告投放、线上品类试点或通勤试错行动题。",
            "题干要求在预算、时间、组织能力、固定目的地、线下口碑或合规风险等约束下选择可执行下一步。",
            "题目不是深技术迁移、API 架构治理或安全攻击面主导的问题；这类题先不触发该后代。",
        ],
        predicted_effects=[
            "保留上一代在四个真实 trigger 上的执行收益，同时补齐上一版 scope-only child 输掉的具体桥接字段。",
            "把抽象的 metric/owner/rollback 清单转成题目特定结构：城市后验、保密合同、分阶段组织激励、P95 实测。",
            "把上一代 software migration 负例从当前后代作用域中移出，避免把未修复残差继续写进主图。",
            "为后续 technical-migration child 提供清晰 residual target，而不是让一个泛化后代包打所有执行题。",
        ],
        risk_predictions=[
            "如果只做收窄而不产生新能力，论文里应把它解释为 selective retention，不应夸成泛化提升。",
            "如果下一代答案过于模板化，仍可能输给自然 baseline；必须用同模型 live judge 再验。",
        ],
        verifiers=[
            "same_model_trigger_answer_quality",
            "route_scoped_noop_control_check",
            "live_residual_abstain_check",
            "bounded_structural_morphism_certificate_check",
        ],
        residual_ids=["res_software_migration_scope_loss_20260608"],
        confidence=0.55,
        metaproductivity=0.18,
        status="candidate",
        tags=[
            "descendant",
            "execution_contract",
            "constraint_bridge",
            "selective_retention",
            "abstain_boundary",
            "residual_repair",
            "residual_specific_bridge",
            "proof_lite_diagram",
            "generation_4",
            "live_repair_v4",
        ],
        source_refs=[
            "orthogonal_descendant_live_same_model_20260608",
            "orthogonal_descendant_nextgen_live_same_model_20260609_failed_scope_only_child",
            f"parent:{PARENT_CANDIDATE_ID}",
            f"abstained_residual:{ABSTAINED_RESIDUAL_IDS[0]}",
        ],
        payload={
            "activation": {
                "problem_ids": TRIGGER_IDS,
                "min_keyword_hits": 1,
                "allow_lexical_fallback": False,
            },
            "descendant_generation": 4,
            "parent_family": "orthogonal_execution_contract",
            "parent_candidate_node_id": PARENT_CANDIDATE_ID,
            "parent_proposal_id": PARENT_PROPOSAL_ID,
            "live_parent_outcome": {
                "trigger_wins": 4,
                "trigger_losses": 1,
                "control_ties": 8,
                "lost_trigger_problem_ids": ABSTAINED_RESIDUAL_IDS,
            },
            "failed_child_residuals": {
                "business_0097": [
                    "city/channel posterior update",
                    "separate conversion and brand metrics",
                    "marginal efficiency decay after scale-up",
                ],
                "business_0192": [
                    "layered access for proprietary secrecy",
                    "contract and delivery structure",
                    "timeline and stop-loss decision point",
                ],
                "business_0218": [
                    "two-year staged transition path",
                    "course/city segmentation",
                    "role-aligned incentives and rollback gates",
                ],
            },
            "variation_evaluation_retention": {
                "variation": "residual-specific bridge child generated from the accepted v2 descendant and failed scope-only child judgments",
                "evaluation": "same-model live trigger answers, pairwise judge, route-scoped no-op controls",
                "selective_retention": "retain only the live-positive scope; add concrete residual bridges; abstain on un-repaired software migration residual",
            },
        },
    )
    edge = AssumptionEdge(
        source=candidate_id,
        target=PARENT_CANDIDATE_ID,
        type=EdgeType.SPECIALIZES,
        weight=0.78,
        evidence="orthogonal_descendant_live_same_model_20260608",
        payload={
            "source": "orthogonal_descendant_nextgen_live_queue",
            "reason": "generation-4 selective-retention descendant of the accepted constraint bridge",
            "abstained_residual_problem_ids": ABSTAINED_RESIDUAL_IDS,
        },
    )
    return {
        "eval_id": eval_id,
        "source_eval_id": "orthogonal_descendant_nextgen_live_queue_builder",
        "proposal_counts": {"descendant_execution_hypothesis": 1},
        "proposals": [{
            "proposal_id": proposal_id,
            "proposal_type": "descendant_execution_hypothesis",
            "parent_node_id": PARENT_CANDIDATE_ID,
            "candidate_node": candidate.to_dict(),
            "edges": [edge.to_dict()],
            "manifest": None,
            "rationale": (
                "The accepted v2 descendant won four trigger rows but lost the software migration row. "
                "This child implements variation/evaluation/selective retention by keeping the live-positive "
                "scope and abstaining on the observed residual."
            ),
            "priority": 0.82,
            "source_action": {
                "action_type": "export_next_generation_descendant_for_live_ablation",
                "descendant_generation": 4,
                "repair_variant": "residual_specific_retained_scope_v4",
                "parent_node_id": PARENT_CANDIDATE_ID,
                "abstained_residual_problem_ids": ABSTAINED_RESIDUAL_IDS,
            },
        }],
    }


def _prepare_nextgen_graph_snapshot(
    *,
    source_graph_dir: Path,
    retained_graph_dir: Path,
    parent_queue_path: Path,
) -> dict[str, Any]:
    _copy_graph(source_graph_dir, retained_graph_dir)
    parent_queue = _load_json(parent_queue_path)
    parent_payload = parent_queue.get("proposal_payload", parent_queue)
    parent_proposal = _find_proposal(parent_payload, PARENT_PROPOSAL_ID)
    parent_node = AssumptionNode.from_dict(parent_proposal["candidate_node"])
    parent_node.status = "active"

    source_store = JsonlGraphStore(source_graph_dir)
    store = JsonlGraphStore(retained_graph_dir)
    before_node_count = len(store.nodes)
    before_edge_count = len(store.edges)
    seed_present_before = SEED_CANDIDATE_ID in store.nodes
    source_graph_had_parent = parent_node.id in source_store.nodes

    store.upsert_node(parent_node)
    for edge in parent_proposal.get("edges", []):
        store.add_edge(AssumptionEdge.from_dict(edge))
    store.flush()

    after = JsonlGraphStore(retained_graph_dir)
    parent_after = after.nodes.get(parent_node.id)
    edge_counts = Counter(
        str(edge.type.value if hasattr(edge.type, "value") else edge.type)
        for edge in after.edges
        if edge.source == parent_node.id
    )
    return {
        "path": str(retained_graph_dir),
        "seed_candidate_node_id": SEED_CANDIDATE_ID,
        "seed_present_before_parent_snapshot": seed_present_before,
        "parent_proposal_id": parent_proposal.get("proposal_id"),
        "parent_candidate_node_id": parent_node.id,
        "parent_status_after_snapshot": str(getattr(parent_after, "status", "")) if parent_after else "",
        "parent_parent_node_id": parent_proposal.get("parent_node_id"),
        "source_graph_had_parent": source_graph_had_parent,
        "base_node_count": before_node_count,
        "base_edge_count": before_edge_count,
        "snapshot_node_count": len(after.nodes),
        "snapshot_edge_count": len(after.edges),
        "parent_edge_counts": dict(edge_counts),
        "source_snapshot_mutated": parent_node.id in JsonlGraphStore(source_graph_dir).nodes,
    }


def _find_proposal(payload: dict[str, Any], proposal_id: str) -> dict[str, Any]:
    for proposal in payload.get("proposals", []):
        if proposal.get("proposal_id") == proposal_id:
            return proposal
    raise ValueError(f"Proposal {proposal_id} not found")


def _evolution_payload(
    *,
    eval_id: str,
    proposal_payload: dict[str, Any],
    preflight: dict[str, Any],
    novelty: dict[str, Any],
) -> dict[str, Any]:
    summaries = preflight.get("summaries", [])
    return {
        "eval_id": f"{eval_id}_evolution",
        "proposals": proposal_payload,
        "candidate_preflight": preflight,
        "novelty_integration": novelty,
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
                    "expected_value": 0.64,
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
                    "retention_boundary": "live_positive_scope_only",
                }
                for row in summaries
            ],
        },
        "regression_predictions": [
            {
                "proposal_id": row.get("proposal_id"),
                "risk": "scope_narrowing_may_not_generalize_but_should_reduce_observed_residual_harm",
            }
            for row in summaries
        ],
        "formal_mapping_gate": {
            "gates": [{
                "gate": "bounded_structural_morphism_proof_lite",
                "status": "recorded",
                "invariants": proposal_payload["proposals"][0]["candidate_node"].get("formal_form", {})
                .get("category_inspired_diagram", {})
                .get("invariants", []),
            }],
        },
    }


def _fixture_readback(
    *,
    root: Path,
    graph_dir: Path,
    proposal_payload: dict[str, Any],
    preflight: dict[str, Any],
    evolution: dict[str, Any],
    novelty: dict[str, Any],
    eval_id: str,
) -> dict[str, Any]:
    proposal_id = proposal_payload["proposals"][0]["proposal_id"]
    candidate_id = proposal_payload["proposals"][0]["candidate_node"]["id"]
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        temp_graph = temp_root / "graph"
        _copy_graph(graph_dir, temp_graph)
        before_nodes = set(JsonlGraphStore(temp_graph).nodes)
        judgment_sets = _write_fixture_judgment_sets(temp_root, preflight, proposal_id)
        readback = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight,
            evolution_payload=evolution,
            judgment_sets=judgment_sets,
            eval_id=f"{eval_id}_readback",
            queue_name="orthogonal_descendant_nextgen_live_queue",
            command_limit=1,
            execute=False,
            apply_accepted=False,
            writeback_manifests=True,
        )
        after_readback_nodes = set(JsonlGraphStore(temp_graph).nodes)
        applied = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight,
            evolution_payload=evolution,
            judgment_sets=judgment_sets,
            eval_id=f"{eval_id}_apply",
            queue_name="orthogonal_descendant_nextgen_live_queue",
            command_limit=1,
            execute=False,
            apply_accepted=True,
            writeback_manifests=True,
        )
        applied_store = JsonlGraphStore(temp_graph)
        candidate_present = candidate_id in applied_store.nodes
        edge_counts = Counter(
            str(edge.type.value if hasattr(edge.type, "value") else edge.type)
            for edge in applied_store.edges
            if edge.source == candidate_id
        )
    return {
        "evolution_payload": evolution,
        "fixture_readback": _compact_daemon(readback),
        "fixture_temp_apply": _compact_daemon(applied),
        "node_mutation_without_apply": before_nodes != after_readback_nodes,
        "candidate_node_present_after_apply": candidate_present,
        "candidate_edge_counts_after_apply": dict(edge_counts),
        "novelty_integration": novelty,
    }


def _write_fixture_judgment_sets(temp_root: Path, preflight: dict[str, Any], proposal_id: str) -> list[JudgmentSet]:
    baseline_variant = "phase2_v20_claude_opus_execution_baseline"
    candidate_variant = f"proposal_{proposal_id.replace('prop_', '')}"
    summary = {row["proposal_id"]: row for row in preflight.get("summaries", [])}[proposal_id]
    rows: dict[str, dict[str, str]] = {}
    for pid in summary.get("trigger_problem_ids", []):
        rows[pid] = {"winner": candidate_variant}
    for pid in summary.get("control_problem_ids", []):
        rows[pid] = {"winner": "tie", "a_was": candidate_variant, "b_was": baseline_variant}
    path = temp_root / f"{proposal_id}_fixture_judgments.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [JudgmentSet(
        candidate_variant=candidate_variant,
        baseline_variant=baseline_variant,
        judgment_paths=[path],
        proposal_ids=[proposal_id],
    )]


def _metrics(
    *,
    proposal_payload: dict[str, Any],
    novelty: dict[str, Any],
    preflight: dict[str, Any],
    readback: dict[str, Any],
    retained_graph: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    summary = preflight.get("summaries", [{}])[0]
    novelty_counts = novelty.get("classification_counts", {})
    edge_counts = novelty.get("recommended_edge_counts", {})
    readback_counts = readback.get("fixture_readback", {}).get("candidate_acceptance_counts", {})
    apply_counts = readback.get("fixture_temp_apply", {}).get("candidate_acceptance_counts", {})
    return {
        "proposal_count": len(proposal_payload.get("proposals", [])),
        "specialization_count": int(novelty_counts.get("specialization", 0)),
        "specializes_edge_count": int(edge_counts.get(EdgeType.SPECIALIZES.value, 0)),
        "preflight_ready_count": sum(1 for row in preflight.get("summaries", []) if row.get("readiness") == "ready_for_fresh_ablation"),
        "trigger_count": len(summary.get("trigger_problem_ids", [])),
        "active_trigger_count": len(summary.get("active_trigger_problem_ids", [])),
        "trigger_problem_ids": summary.get("trigger_problem_ids", []),
        "control_count": len(summary.get("control_problem_ids", [])),
        "control_problem_ids": summary.get("control_problem_ids", []),
        "outside_active_count": len(summary.get("outside_active_problem_ids", [])),
        "outside_active_problem_ids": summary.get("outside_active_problem_ids", []),
        "abstained_residual_problem_ids": ABSTAINED_RESIDUAL_IDS,
        "parent_snapshot_specializes_count": int(retained_graph["parent_edge_counts"].get(EdgeType.SPECIALIZES.value, 0)),
        "readback_accept_count": int(readback_counts.get("accept", 0)),
        "readback_applied_count": len(readback.get("fixture_readback", {}).get("applied_candidate_node_ids", [])),
        "apply_accept_count": int(apply_counts.get("accept", 0)),
        "apply_applied_count": len(readback.get("fixture_temp_apply", {}).get("applied_candidate_node_ids", [])),
        "node_mutation_without_apply": bool(readback.get("node_mutation_without_apply")),
        "candidate_node_present_after_apply": bool(readback.get("candidate_node_present_after_apply")),
        "live_env_ready": bool(env["solver_ready"] and env["judge_ready"]),
    }


def _next_commands(root: Path, proposal_id: str, proposals_out: Path, preflight_out: Path) -> list[dict[str, str]]:
    queue_path = DEFAULT_OUT
    out_path = PAPER_DIR / "orthogonal_descendant_nextgen_live_same_model_20260609.json"
    readback_path = PAPER_DIR / "orthogonal_descendant_nextgen_live_readback_20260609.json"
    return [
        {
            "name": "run_nextgen_descendant_live_same_model",
            "command": (
                "LLM_PROVIDER=gpt GPT5_API_KEY=<set-in-env> GPT5_BASE_URL=<set-in-env> "
                "GPT5_MODEL=claude-opus-4-8 RUOLI_CLAUDE_KEY=<set-in-env> "
                "CLAUDE_BASE_URL=<set-in-env> CLAUDE_OPUS_MODEL=claude-opus-4-8 "
                "python3 -m assumption_os.orthogonal_live_ablation --root . "
                f"--queue '{_display_path(root, root / queue_path)}' "
                "--eval-id orthogonal_descendant_nextgen_live_same_model_20260609 "
                "--execute-answers --run-judge --judge-model claude_opus "
                "--baseline-variant phase2_v20_claude_opus_execution_baseline "
                "--route-scoped-noop-controls "
                f"--out '{_display_path(root, root / out_path)}'"
            ),
        },
        {
            "name": "read_back_nextgen_live_acceptance",
            "command": (
                "python3 -m assumption_os.orthogonal_descendant_live_readback --root . "
                f"--queue '{_display_path(root, root / queue_path)}' "
                f"--live '{_display_path(root, root / out_path)}' "
                "--eval-id orthogonal_descendant_nextgen_live_readback_20260609 "
                f"--out '{_display_path(root, root / readback_path)}'"
            ),
        },
        {
            "name": "proposal_payload",
            "command": _display_path(root, proposals_out),
        },
        {
            "name": "preflight_payload",
            "command": _display_path(root, preflight_out),
        },
        {
            "name": "proposal_id",
            "command": proposal_id,
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build next-generation live queue for an accepted orthogonal descendant.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--source-graph-dir", default=str(DEFAULT_SOURCE_GRAPH_DIR))
    ap.add_argument("--parent-queue", default=str(DEFAULT_PARENT_QUEUE))
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    ap.add_argument("--meta", default=str(DEFAULT_META_PATH))
    ap.add_argument("--eval-id", default="orthogonal_descendant_nextgen_live_queue_20260609")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--proposals-out", default=str(DEFAULT_PROPOSALS_OUT))
    ap.add_argument("--preflight-out", default=str(DEFAULT_PREFLIGHT_OUT))
    ap.add_argument("--retained-graph-dir", default=str(DEFAULT_RETAINED_GRAPH_DIR))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_orthogonal_descendant_nextgen_live_queue_payload(
        root=root,
        source_graph_dir=Path(args.source_graph_dir),
        parent_queue_path=Path(args.parent_queue),
        sample_path=Path(args.sample),
        meta_path=Path(args.meta),
        eval_id=args.eval_id,
        proposals_out=Path(args.proposals_out),
        preflight_out=Path(args.preflight_out),
        retained_graph_dir=Path(args.retained_graph_dir),
    )
    out = _resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "status": payload["status"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": _display_path(root, out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
