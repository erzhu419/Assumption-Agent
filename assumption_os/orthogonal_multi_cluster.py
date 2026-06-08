"""Multi-cluster validation for orthogonal hypothesis integration.

The single positive queue proves that one hand-built orthogonal candidate can
pass novelty/preflight/readback.  This module raises that bar: it creates
several orthogonal candidates over different residual/parent families, then
validates ON/OFF novelty behavior, fresh-ablation preflight, daemon readback,
and gated temporary apply without requiring live API credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_eval import CandidateReadiness, build_candidate_eval_payload
from .graph_memory import JsonlGraphStore
from .novelty_integration import NoveltyClass, build_novelty_integration_payload
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


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_GRAPH_DIR = Path("phase four/assumption_graph")
DEFAULT_SAMPLE_PATH = Path("phase two/analysis/cache/sample_100.json")
DEFAULT_META_PATH = Path("phase two/analysis/cache/answers/phase2_v20_meta.json")
DEFAULT_OUT = PAPER_DIR / "orthogonal_multi_cluster_20260608.json"
DEFAULT_PROPOSALS_OUT = PAPER_DIR / "orthogonal_multi_cluster_proposals_20260608.json"
DEFAULT_PREFLIGHT_OUT = PAPER_DIR / "orthogonal_multi_cluster_preflight_20260608.json"


@dataclass(frozen=True)
class OrthogonalCandidateSpec:
    parent_id: str
    axis: str
    node_type: AssumptionType
    kind: HypothesisKind
    claim: str
    context_conditions: list[str]
    predicted_effects: list[str]
    risk_predictions: list[str]
    verifiers: list[str]
    tags: list[str]
    residual_id: str
    trigger_problem_ids: list[str]
    rationale: str
    priority: float


MULTI_CLUSTER_SPECS = [
    OrthogonalCandidateSpec(
        parent_id="strategy_S01",
        axis="rubric_drift",
        node_type=AssumptionType.EVALUATOR,
        kind=HypothesisKind.EVALUATOR_POLICY,
        claim=(
            "Before repairing the task method, test whether cross-judge rubric drift or stale acceptance "
            "feedback caused the residual; calibrate the evaluator axis against trigger and placebo controls."
        ),
        context_conditions=[
            "same failure residual as a method repair candidate",
            "answer content looks plausible but acceptance flips across judge/model/rubric variants",
        ],
        predicted_effects=[
            "avoid promoting method edits when the failure is caused by evaluator drift",
            "improve recursive retention by separating evaluator defects from method defects",
        ],
        risk_predictions=[
            "may waste ablation budget if the residual is actually a method defect",
            "may overfit to judge phrasing unless placebo controls pass",
        ],
        verifiers=[
            "cross_judge_disagreement_probe",
            "trigger_control_fresh_ablation",
            "placebo_rubric_stability_check",
        ],
        tags=[
            "candidate",
            "orthogonal",
            "rubric_drift",
            "cross_judge_calibration",
            "acceptance_noise",
        ],
        residual_id="res_orthogonal_multi_cluster_rubric_drift",
        trigger_problem_ids=[
            "business_0097",
            "engineering_0244",
            "daily_life_0173",
            "software_engineering_0086",
        ],
        rationale=(
            "The same residual may be explained by evaluator drift rather than the controlled-variable strategy."
        ),
        priority=0.72,
    ),
    OrthogonalCandidateSpec(
        parent_id="strategy_S25",
        axis="simulator_gap",
        node_type=AssumptionType.WORLD_MODEL,
        kind=HypothesisKind.CLAIM,
        claim=(
            "Before editing the reasoning recipe, test whether the missing scale-calibrated surrogate model caused "
            "the residual; compare component-level evidence with aggregate behavior under a cheap simulator."
        ),
        context_conditions=[
            "component observations look valid but aggregate behavior diverges after interaction",
            "the answer needs a calibrated surrogate before committing to a policy recommendation",
        ],
        predicted_effects=[
            "separate simulator defects from reasoning-strategy defects",
            "produce cheaper falsification tests before spending live model calls on broad repairs",
        ],
        risk_predictions=[
            "the surrogate can become misleading if its boundary conditions are not recorded",
            "simulation probes may delay useful method repairs when direct evidence is already decisive",
        ],
        verifiers=[
            "scale_surrogate_probe",
            "micro_macro_invariant_check",
            "negative_control_component_aggregation",
        ],
        tags=[
            "candidate",
            "orthogonal",
            "simulator_gap",
            "scale_calibration",
            "surrogate_model",
        ],
        residual_id="res_orthogonal_multi_cluster_simulator_gap",
        trigger_problem_ids=[
            "engineering_0183",
            "science_0175",
            "science_0097",
            "software_engineering_0111",
        ],
        rationale=(
            "The same residual may require a simulator/world-model axis rather than another emergence-rule edit."
        ),
        priority=0.69,
    ),
    OrthogonalCandidateSpec(
        parent_id="strategy_S26",
        axis="provenance_archive_gap",
        node_type=AssumptionType.MEMORY,
        kind=HypothesisKind.CLAIM,
        claim=(
            "Provenance-archive gap: stale source ledgers hide prior commitments; retrieve source anchors before "
            "intervention."
        ),
        context_conditions=[
            "source-ledger mismatch",
            "hidden commitment evidence unavailable in the prompt",
        ],
        predicted_effects=[
            "separate memory defect from method defect",
            "preserve source-ledger memory for later proposals",
        ],
        risk_predictions=[
            "irrelevant archive retrieval without source anchors",
            "over-conservative answer if present constraints are ignored",
        ],
        verifiers=[
            "decision_ledger_retrieval_probe",
            "source_anchor_control",
            "stale_context_negative_control",
        ],
        tags=[
            "candidate",
            "orthogonal",
            "provenance_archive",
            "source_ledger",
            "stale_record",
        ],
        residual_id="res_orthogonal_multi_cluster_provenance_archive_gap",
        trigger_problem_ids=[
            "business_0192",
            "software_engineering_0142",
            "business_0218",
            "software_engineering_0364",
        ],
        rationale=(
            "The same residual may be a provenance-memory defect rather than a path-dependency strategy defect."
        ),
        priority=0.68,
    ),
]


def build_orthogonal_multi_cluster_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    sample_path: Path | None = None,
    meta_path: Path | None = None,
    eval_id: str | None = None,
    proposals_out: Path | None = None,
    preflight_out: Path | None = None,
) -> dict[str, Any]:
    """Build the multi-cluster orthogonal validation payload."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    sample_path = _resolve(root, sample_path or DEFAULT_SAMPLE_PATH)
    meta_path = _resolve(root, meta_path or DEFAULT_META_PATH)
    proposals_out = _resolve(root, proposals_out or DEFAULT_PROPOSALS_OUT)
    preflight_out = _resolve(root, preflight_out or DEFAULT_PREFLIGHT_OUT)
    eval_id = eval_id or "orthogonal_multi_cluster_20260608"

    proposal_payload = _build_multi_cluster_proposal_payload(eval_id=eval_id)
    proposal_ids = [p["proposal_id"] for p in proposal_payload["proposals"]]
    store = JsonlGraphStore(graph_dir)
    novelty_enabled = build_novelty_integration_payload(
        store,
        proposal_payload,
        eval_id=f"{eval_id}_novelty_enabled",
        enable_orthogonal=True,
    )
    novelty_disabled = build_novelty_integration_payload(
        store,
        proposal_payload,
        eval_id=f"{eval_id}_novelty_disabled",
        enable_orthogonal=False,
    )
    preflight = build_candidate_eval_payload(
        graph_dir=graph_dir,
        proposal_payload=proposal_payload,
        sample=_load_json(sample_path),
        meta_by_pid=_load_json(meta_path),
        eval_id=f"{eval_id}_preflight",
        proposal_ids=proposal_ids,
        min_trigger_n=3,
        min_active_trigger_n=3,
        force_proposal_route=True,
        proposals_arg=_display_path(root, proposals_out),
        sample_arg=_display_path(root, sample_path),
    )
    readback = _validate_daemon_readback(
        root=root,
        graph_dir=graph_dir,
        proposal_payload=proposal_payload,
        preflight=preflight,
        novelty_enabled=novelty_enabled,
        eval_id=eval_id,
    )
    env = _env_status()
    next_commands = _next_commands(
        root=root,
        preflight=preflight,
        proposals_out=proposals_out,
        preflight_out=preflight_out,
    )
    metrics = _metrics(
        proposal_payload=proposal_payload,
        novelty_enabled=novelty_enabled,
        novelty_disabled=novelty_disabled,
        preflight=preflight,
        readback=readback,
        env=env,
    )
    gates = {
        "has_three_distinct_parent_clusters": metrics["distinct_parent_count"] >= 3,
        "orthogonal_enabled_classifies_all_new_families": (
            metrics["enabled_orthogonal_count"] == metrics["proposal_count"]
        ),
        "orthogonal_disabled_removes_orthogonal_class": metrics["disabled_orthogonal_count"] == 0,
        "orthogonal_edges_only_when_enabled": (
            metrics["enabled_orthogonal_edge_count"] == metrics["proposal_count"]
            and metrics["disabled_orthogonal_edge_count"] == 0
        ),
        "all_candidates_ready_for_fresh_ablation": metrics["preflight_ready_count"] == metrics["proposal_count"],
        "each_candidate_has_trigger_rows": metrics["min_trigger_count"] >= 3,
        "each_candidate_has_active_trigger_rows": metrics["min_active_trigger_count"] >= 3,
        "each_candidate_has_control_rows": metrics["min_control_count"] >= 3,
        "no_no_fire_exposure": metrics["outside_active_total"] == 0,
        "daemon_dry_run_plans_all_ready_leaves": (
            metrics["dry_planned_leaf_count"] == metrics["proposal_count"]
            and metrics["dry_executable_leaf_count"] == metrics["proposal_count"]
            and metrics["dry_status_counts"].get("planned") == metrics["proposal_count"]
        ),
        "fixture_judgments_accept_all_candidates": (
            metrics["readback_accept_count"] == metrics["proposal_count"]
            and metrics["apply_accept_count"] == metrics["proposal_count"]
        ),
        "readback_without_apply_does_not_mutate_graph": (
            metrics["readback_applied_count"] == 0 and not metrics["node_mutation_without_apply"]
        ),
        "gated_temp_apply_writes_all_candidates": metrics["apply_applied_count"] == metrics["proposal_count"],
        "gated_temp_apply_writes_orthogonal_edges": (
            metrics["temp_orthogonal_edge_count"] >= metrics["proposal_count"]
        ),
        "commands_are_secret_free": _commands_are_secret_free(next_commands),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_multi_cluster_live_ready_bridge",
        "performance_validation": True,
        "validation_scope": (
            "multi-parent orthogonal-new-family novelty, trigger/control preflight, daemon readback, "
            "and gated temporary apply using fixture judgments; live answer-quality wins still require API "
            "environment variables and fresh model/judge calls"
        ),
        "status": "multi_cluster_live_ready" if metrics["live_env_ready"] else "multi_cluster_live_ready_env_missing",
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "sample_path": _display_path(root, sample_path),
            "meta_path": _display_path(root, meta_path),
            "proposals_out": _display_path(root, proposals_out),
            "preflight_out": _display_path(root, preflight_out),
        },
        "proposal_payload": proposal_payload,
        "novelty_enabled_summary": _condition_summary(novelty_enabled),
        "novelty_disabled_summary": _condition_summary(novelty_disabled),
        "novelty_rows": {
            "enabled": novelty_enabled.get("rows", []),
            "disabled": novelty_disabled.get("rows", []),
        },
        "preflight_payload": preflight,
        "daemon_validation": readback,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "env_status": env,
        "next_commands": next_commands,
        "interpretation": (
            "Orthogonal novelty is no longer backed by a single example.  Three distinct residual/parent "
            "families now produce low-overlap candidate hypotheses that ON/OFF classify correctly, have "
            "trigger/control rows, can be consumed by the recursive daemon, and are only written through gated "
            "acceptance.  This validates the mechanism; it does not substitute for live downstream QA/answer "
            "quality ablation."
        ),
    }


def _build_multi_cluster_proposal_payload(*, eval_id: str) -> dict[str, Any]:
    proposals = [_proposal_from_spec(eval_id=eval_id, spec=spec) for spec in MULTI_CLUSTER_SPECS]
    return {
        "eval_id": eval_id,
        "source_eval_id": "orthogonal_multi_cluster_builder",
        "proposal_counts": {"orthogonal_failure_hypothesis": len(proposals)},
        "proposals": proposals,
    }


def _proposal_from_spec(*, eval_id: str, spec: OrthogonalCandidateSpec) -> dict[str, Any]:
    candidate_id = stable_id("cand", eval_id, spec.parent_id, spec.axis)
    proposal_id = stable_id("prop", eval_id, candidate_id)
    candidate = AssumptionNode(
        id=candidate_id,
        type=spec.node_type,
        kind=spec.kind,
        claim=spec.claim,
        context_conditions=spec.context_conditions,
        predicted_effects=spec.predicted_effects,
        risk_predictions=spec.risk_predictions,
        verifiers=spec.verifiers,
        residual_ids=[spec.residual_id],
        confidence=0.40,
        metaproductivity=0.05,
        status="candidate",
        tags=spec.tags,
        source_refs=[
            f"parent:{spec.parent_id}",
            "orthogonal_multi_cluster",
            "reconstruction/md/orthogonal_hypothesis_gate_20260608.md",
        ],
        payload={
            "orthogonal_to_existing": True,
            "activation": {
                "problem_ids": spec.trigger_problem_ids,
                "min_keyword_hits": 1,
            },
            "proposal_type": "orthogonal_failure_hypothesis",
            "orthogonal_axis": spec.axis,
            "residual_cluster": spec.residual_id,
            "variation_evaluation_retention": {
                "variation": "new explanatory axis for the same residual context",
                "evaluation": "novelty ON/OFF, trigger/control preflight, daemon readback, and fresh ablation",
                "selective_retention": "retain only if downstream judged triggers clear controls",
            },
        },
    )
    edge = AssumptionEdge(
        source=candidate_id,
        target=spec.parent_id,
        type=EdgeType.GENERATED_FROM_RESIDUAL,
        weight=0.52,
        evidence="orthogonal_multi_cluster",
        payload={
            "source": "orthogonal_multi_cluster",
            "reason": "same residual/parent, different explanatory axis",
            "orthogonal_axis": spec.axis,
        },
    )
    return {
        "proposal_id": proposal_id,
        "proposal_type": "orthogonal_failure_hypothesis",
        "parent_node_id": spec.parent_id,
        "candidate_node": candidate.to_dict(),
        "edges": [edge.to_dict()],
        "manifest": None,
        "rationale": spec.rationale,
        "priority": spec.priority,
        "source_action": {
            "action_type": "orthogonal_multi_cluster",
            "parent_node_id": spec.parent_id,
            "orthogonal_axis": spec.axis,
            "trigger_problem_ids": spec.trigger_problem_ids,
        },
    }


def _validate_daemon_readback(
    *,
    root: Path,
    graph_dir: Path,
    proposal_payload: dict[str, Any],
    preflight: dict[str, Any],
    novelty_enabled: dict[str, Any],
    eval_id: str,
) -> dict[str, Any]:
    evolution = _evolution_payload(
        eval_id=eval_id,
        proposal_payload=proposal_payload,
        preflight=preflight,
        novelty_enabled=novelty_enabled,
    )
    proposal_ids = [p["proposal_id"] for p in proposal_payload.get("proposals", [])]
    candidate_ids = {
        p["proposal_id"]: p.get("candidate_node", {}).get("id")
        for p in proposal_payload.get("proposals", [])
    }
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        temp_graph = temp_root / "graph"
        _copy_graph(graph_dir, temp_graph)
        before_nodes = set(JsonlGraphStore(temp_graph).nodes)
        judgment_sets = _write_fixture_judgment_sets(
            temp_root=temp_root,
            preflight=preflight,
            proposal_ids=proposal_ids,
        )
        dry = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight,
            evolution_payload=evolution,
            eval_id=f"{eval_id}_dry",
            queue_name="orthogonal_multi_cluster",
            command_limit=len(proposal_ids),
            execute=False,
            apply_accepted=False,
            writeback_manifests=True,
        )
        after_dry_nodes = set(JsonlGraphStore(temp_graph).nodes)
        readback = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight,
            evolution_payload=evolution,
            judgment_sets=judgment_sets,
            eval_id=f"{eval_id}_readback",
            queue_name="orthogonal_multi_cluster",
            command_limit=len(proposal_ids),
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
            queue_name="orthogonal_multi_cluster",
            command_limit=len(proposal_ids),
            execute=False,
            apply_accepted=True,
            writeback_manifests=True,
        )
        applied_store = JsonlGraphStore(temp_graph)
        applied_nodes = set(applied_store.nodes)
        temp_candidate_node_count = sum(
            1 for pid in proposal_ids if candidate_ids.get(pid) in applied_nodes
        )
        temp_orthogonal_edge_count = sum(
            _orthogonal_edge_count(applied_store, str(candidate_ids.get(pid) or ""))
            for pid in proposal_ids
        )
    return {
        "evolution_payload": evolution,
        "daemon_dry_run": _compact_daemon_payload(dry),
        "fixture_readback": _compact_daemon_payload(readback),
        "fixture_temp_apply": _compact_daemon_payload(applied),
        "node_mutation_without_apply": before_nodes != after_dry_nodes or before_nodes != after_readback_nodes,
        "temp_candidate_node_count": temp_candidate_node_count,
        "temp_orthogonal_edge_count": temp_orthogonal_edge_count,
    }


def _evolution_payload(
    *,
    eval_id: str,
    proposal_payload: dict[str, Any],
    preflight: dict[str, Any],
    novelty_enabled: dict[str, Any],
) -> dict[str, Any]:
    summaries = preflight.get("summaries", [])
    return {
        "eval_id": f"{eval_id}_evolution",
        "proposals": proposal_payload,
        "candidate_preflight": preflight,
        "novelty_integration": novelty_enabled,
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
                    "expected_value": 0.5,
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
                "risk": "requires_live_controls",
            }
            for row in summaries
        ],
        "formal_mapping_gate": {"gates": []},
    }


def _write_fixture_judgment_sets(
    *,
    temp_root: Path,
    preflight: dict[str, Any],
    proposal_ids: list[str],
) -> list[JudgmentSet]:
    baseline_variant = "phase2_v20_gpt54mini_prop_union"
    summary_by_id = {row["proposal_id"]: row for row in preflight.get("summaries", [])}
    out = []
    for proposal_id in proposal_ids:
        candidate_variant = f"proposal_{proposal_id.replace('prop_', '')}"
        rows: dict[str, dict[str, str]] = {}
        summary = summary_by_id.get(proposal_id, {})
        for pid in summary.get("trigger_problem_ids", []):
            rows[pid] = {"winner": candidate_variant}
        for pid in summary.get("control_problem_ids", []):
            rows[pid] = {
                "winner": "tie",
                "a_was": candidate_variant,
                "b_was": baseline_variant,
            }
        path = temp_root / f"{proposal_id}_fixture_judgments.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out.append(JudgmentSet(
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            judgment_paths=[path],
            proposal_ids=[proposal_id],
        ))
    return out


def _metrics(
    *,
    proposal_payload: dict[str, Any],
    novelty_enabled: dict[str, Any],
    novelty_disabled: dict[str, Any],
    preflight: dict[str, Any],
    readback: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    summaries = preflight.get("summaries", [])
    proposal_count = len(proposal_payload.get("proposals", []))
    trigger_counts = [len(row.get("trigger_problem_ids", [])) for row in summaries]
    active_counts = [len(row.get("active_trigger_problem_ids", [])) for row in summaries]
    control_counts = [len(row.get("control_problem_ids", [])) for row in summaries]
    outside_counts = [len(row.get("outside_active_problem_ids", [])) for row in summaries]
    dry = readback.get("daemon_dry_run", {})
    fixture = readback.get("fixture_readback", {})
    applied = readback.get("fixture_temp_apply", {})
    return {
        "proposal_count": proposal_count,
        "distinct_parent_count": len({
            p.get("parent_node_id")
            for p in proposal_payload.get("proposals", [])
            if p.get("parent_node_id")
        }),
        "enabled_orthogonal_count": novelty_enabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "disabled_orthogonal_count": novelty_disabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "enabled_orthogonal_edge_count": novelty_enabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "disabled_orthogonal_edge_count": novelty_disabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "disabled_specialization_count": novelty_disabled.get("classification_counts", {}).get(
            NoveltyClass.SPECIALIZATION.value,
            0,
        ),
        "preflight_ready_count": preflight.get("readiness_counts", {}).get(
            CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
            0,
        ),
        "min_trigger_count": min(trigger_counts) if trigger_counts else 0,
        "min_active_trigger_count": min(active_counts) if active_counts else 0,
        "min_control_count": min(control_counts) if control_counts else 0,
        "outside_active_total": sum(outside_counts),
        "dry_planned_leaf_count": dry.get("planned_leaf_count", 0),
        "dry_executable_leaf_count": dry.get("executable_leaf_count", 0),
        "dry_status_counts": dry.get("execution_status_counts", {}),
        "readback_accept_count": fixture.get("candidate_acceptance_counts", {}).get("accept", 0),
        "readback_resumed": bool(fixture.get("resumed")),
        "readback_applied_count": len(fixture.get("applied_candidate_node_ids") or []),
        "apply_accept_count": applied.get("candidate_acceptance_counts", {}).get("accept", 0),
        "apply_resumed": bool(applied.get("resumed")),
        "apply_applied_count": len(applied.get("applied_candidate_node_ids") or []),
        "temp_candidate_node_count": readback.get("temp_candidate_node_count", 0),
        "temp_orthogonal_edge_count": readback.get("temp_orthogonal_edge_count", 0),
        "node_mutation_without_apply": bool(readback.get("node_mutation_without_apply")),
        "live_env_ready": env["gpt"]["ready"] or env["ruoli_gpt"]["ready"],
    }


def _next_commands(
    *,
    root: Path,
    preflight: dict[str, Any],
    proposals_out: Path,
    preflight_out: Path,
) -> list[dict[str, str]]:
    commands = []
    for row in preflight.get("summaries", []):
        proposal_id = row.get("proposal_id", "")
        answer_command = (
            "RUOLI_GPT_KEY=<set-in-env> RUOLI_BASE_URL=<set-in-env> "
            "GPT_MINI_MODEL=gpt-5.4-mini "
            f"{row.get('command_hint', '')}"
        )
        acceptance_command = (
            "python3 -m assumption_os.candidate_acceptance "
            f"--root . --proposals '{_display_path(root, proposals_out)}' "
            f"--preflight '{_display_path(root, preflight_out)}' "
            "--judgments '<judgments-json-from-pairwise-judge>' "
            f"--candidate-variant proposal_{str(proposal_id).replace('prop_', '')} "
            "--baseline-variant phase2_v20_gpt54mini_prop_union "
            f"--eval-id acceptance_{proposal_id}_orthogonal_multi_cluster --proposal-ids {proposal_id} "
            "--summary-out '<acceptance-summary-json>'"
        )
        commands.append({
            "proposal_id": proposal_id,
            "name": "fresh_ablation_answers",
            "command": answer_command,
        })
        commands.append({
            "proposal_id": proposal_id,
            "name": "acceptance_gate_after_pairwise_judgments",
            "command": acceptance_command,
        })
    return commands


def _condition_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": payload.get("eval_id"),
        "orthogonal_gate_enabled": payload.get("orthogonal_gate_enabled"),
        "classification_counts": payload.get("classification_counts"),
        "recommended_edge_counts": payload.get("recommended_edge_counts"),
        "pass": payload.get("pass"),
    }


def _compact_daemon_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": payload.get("eval_id"),
        "ready_queue_count": payload.get("ready_queue_count"),
        "planned_leaf_count": payload.get("planned_leaf_count"),
        "executable_leaf_count": payload.get("executable_leaf_count"),
        "execution_status_counts": payload.get("execution_status_counts"),
        "candidate_acceptance_counts": payload.get("candidate_acceptance_counts"),
        "accepted_proposal_ids": payload.get("accepted_proposal_ids"),
        "resumed": payload.get("resumed"),
        "applied_candidate_node_ids": payload.get("applied_candidate_node_ids"),
        "apply_summary": payload.get("apply_summary"),
        "manifest_count": payload.get("manifest_count"),
    }


def _copy_graph(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _orthogonal_edge_count(store: JsonlGraphStore, candidate_id: str) -> int:
    return sum(
        1
        for edge in store.edges
        if edge.source == candidate_id and edge.type == EdgeType.ORTHOGONAL_TO
    )


def _commands_are_secret_free(commands: list[dict[str, str]]) -> bool:
    text = json.dumps(commands, ensure_ascii=False)
    return "sk-" not in text and "newapi_channel_conn" not in text and "<set-in-env>" in text


def _env_status() -> dict[str, Any]:
    specs = {
        "gpt": ["GPT5_API_KEY", "GPT5_BASE_URL"],
        "ruoli_gpt": ["RUOLI_GPT_KEY", "RUOLI_BASE_URL"],
        "gemini": ["GEMINI_API_KEY", "GEMINI_BASE_URL"],
        "claude": ["ANTHROPIC_API_KEY"],
    }
    return {
        name: {
            "required_names": names,
            "set_names": [var for var in names if bool(os.environ.get(var))],
            "ready": all(bool(os.environ.get(var)) for var in names),
        }
        for name, names in specs.items()
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-cluster orthogonal proposal readiness.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--meta", default=str(DEFAULT_META_PATH))
    parser.add_argument("--eval-id", default="orthogonal_multi_cluster_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--proposals-out", default=str(DEFAULT_PROPOSALS_OUT))
    parser.add_argument("--preflight-out", default=str(DEFAULT_PREFLIGHT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    proposals_out = _resolve(root, Path(args.proposals_out))
    preflight_out = _resolve(root, Path(args.preflight_out))
    payload = build_orthogonal_multi_cluster_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        sample_path=Path(args.sample),
        meta_path=Path(args.meta),
        eval_id=args.eval_id,
        proposals_out=proposals_out,
        preflight_out=preflight_out,
    )
    _write_json(proposals_out, payload["proposal_payload"])
    _write_json(preflight_out, payload["preflight_payload"])
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "status": payload["status"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
        "proposals_out": str(proposals_out),
        "preflight_out": str(preflight_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
