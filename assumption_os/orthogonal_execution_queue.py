"""Execution-level orthogonal proposal queue.

The first live orthogonal queue proved that the graph can retain new axes, but
the proposals were mostly meta-diagnostic and did not reliably improve answer
quality.  This queue tests a stronger hypothesis: an execution-contract harness
is orthogonal to the method family, yet directly changes the final answer in a
way the downstream judge can evaluate.
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
DEFAULT_OUT = PAPER_DIR / "orthogonal_execution_queue_20260608.json"
DEFAULT_PROPOSALS_OUT = PAPER_DIR / "orthogonal_execution_queue_proposals_20260608.json"
DEFAULT_PREFLIGHT_OUT = PAPER_DIR / "orthogonal_execution_queue_preflight_20260608.json"


DEFAULT_TRIGGER_IDS = [
    "business_0097",
    "engineering_0244",
    "software_engineering_0086",
    "daily_life_0173",
    "business_0192",
    "software_engineering_0142",
    "business_0218",
    "software_engineering_0364",
]


@dataclass(frozen=True)
class ExecutionCandidateSpec:
    parent_id: str
    axis: str
    trigger_problem_ids: list[str]


EXECUTION_SPEC = ExecutionCandidateSpec(
    parent_id="strategy_S01",
    axis="answer_execution_contract",
    trigger_problem_ids=DEFAULT_TRIGGER_IDS,
)


def build_orthogonal_execution_queue_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    sample_path: Path | None = None,
    meta_path: Path | None = None,
    eval_id: str | None = None,
    proposals_out: Path | None = None,
    preflight_out: Path | None = None,
    trigger_problem_ids: list[str] | None = None,
    scope_note: str | None = None,
) -> dict[str, Any]:
    """Build a live-ready execution-level orthogonal candidate queue."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    sample_path = _resolve(root, sample_path or DEFAULT_SAMPLE_PATH)
    meta_path = _resolve(root, meta_path or DEFAULT_META_PATH)
    proposals_out = _resolve(root, proposals_out or DEFAULT_PROPOSALS_OUT)
    preflight_out = _resolve(root, preflight_out or DEFAULT_PREFLIGHT_OUT)
    eval_id = eval_id or "orthogonal_execution_queue_20260608"

    trigger_problem_ids = trigger_problem_ids or DEFAULT_TRIGGER_IDS
    proposal_payload = _build_execution_proposal_payload(
        eval_id=eval_id,
        trigger_problem_ids=trigger_problem_ids,
        scope_note=scope_note,
    )
    proposal_id = proposal_payload["proposals"][0]["proposal_id"]
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
    min_scope_trigger_n = min(6, len(trigger_problem_ids))
    required_declared_trigger_n = min(8, len(trigger_problem_ids))
    preflight = build_candidate_eval_payload(
        graph_dir=graph_dir,
        proposal_payload=proposal_payload,
        sample=_load_json(sample_path),
        meta_by_pid=_load_json(meta_path),
        eval_id=f"{eval_id}_preflight",
        proposal_ids=[proposal_id],
        min_trigger_n=min_scope_trigger_n,
        min_active_trigger_n=min_scope_trigger_n,
        force_proposal_route=True,
        max_control_ids=8,
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
    summary = preflight["summaries"][0] if preflight.get("summaries") else {}
    metrics = _metrics(
        proposal_payload=proposal_payload,
        novelty_enabled=novelty_enabled,
        novelty_disabled=novelty_disabled,
        preflight=preflight,
        readback=readback,
        env=env,
    )
    next_commands = _next_commands(
        root=root,
        summary=summary,
        proposal_id=proposal_id,
        proposals_out=proposals_out,
        preflight_out=preflight_out,
    )
    gates = {
        "single_execution_candidate": metrics["proposal_count"] == 1,
        "orthogonal_enabled_classifies_new_family": metrics["enabled_orthogonal_count"] == 1,
        "orthogonal_disabled_removes_orthogonal_class": metrics["disabled_orthogonal_count"] == 0,
        "orthogonal_edge_only_when_enabled": (
            metrics["enabled_orthogonal_edge_count"] == 1
            and metrics["disabled_orthogonal_edge_count"] == 0
        ),
        "ready_for_fresh_ablation": metrics["preflight_ready_count"] == 1,
        "declared_trigger_rows_reached": metrics["trigger_count"] >= required_declared_trigger_n,
        "declared_active_trigger_rows_reached": metrics["active_trigger_count"] >= required_declared_trigger_n,
        "control_rows_present": metrics["control_count"] >= 8,
        "no_no_fire_exposure": metrics["outside_active_count"] == 0,
        "daemon_dry_run_plans_leaf": (
            metrics["dry_planned_leaf_count"] == 1
            and metrics["dry_executable_leaf_count"] == 1
            and metrics["dry_status_counts"].get("planned") == 1
        ),
        "fixture_judgment_accepts_candidate": (
            metrics["readback_accept_count"] == 1
            and metrics["apply_accept_count"] == 1
        ),
        "readback_without_apply_does_not_mutate_graph": (
            metrics["readback_applied_count"] == 0
            and not metrics["node_mutation_without_apply"]
        ),
        "gated_temp_apply_writes_candidate_and_orthogonal_edge": (
            metrics["apply_applied_count"] == 1
            and metrics["temp_orthogonal_edge_count"] >= 1
        ),
        "commands_are_secret_free": _commands_are_secret_free(next_commands),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_execution_contract_live_ready_queue",
        "performance_validation": True,
        "validation_scope": (
            "execution-level orthogonal-new-family novelty, expanded trigger/control preflight, daemon "
            "readback, and gated temporary apply using fixture judgments; downstream quality still requires "
            "fresh answer generation and pairwise judge"
        ),
        "status": "execution_queue_live_ready" if metrics["live_env_ready"] else "execution_queue_live_ready_env_missing",
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "sample_path": _display_path(root, sample_path),
            "meta_path": _display_path(root, meta_path),
            "proposals_out": _display_path(root, proposals_out),
            "preflight_out": _display_path(root, preflight_out),
            "trigger_problem_ids": trigger_problem_ids,
            "scope_note": scope_note or "default_broad_execution_contract_scope",
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
            "This candidate treats answer execution as a separate harness axis: it does not replace the "
            "controlled-variable method, but requires the final answer to state reversible action, success "
            "thresholds, stop/rollback criteria, ownership, and next evidence.  It is therefore orthogonal to "
            "the parent method family while being directly testable in downstream answer-quality ablation."
        ),
    }


def _build_execution_proposal_payload(
    *,
    eval_id: str,
    trigger_problem_ids: list[str] | None = None,
    scope_note: str | None = None,
) -> dict[str, Any]:
    spec = ExecutionCandidateSpec(
        parent_id=EXECUTION_SPEC.parent_id,
        axis=EXECUTION_SPEC.axis,
        trigger_problem_ids=trigger_problem_ids or EXECUTION_SPEC.trigger_problem_ids,
    )
    candidate_id = stable_id("cand", eval_id, spec.parent_id, spec.axis)
    proposal_id = stable_id("prop", eval_id, candidate_id)
    candidate = AssumptionNode(
        id=candidate_id,
        type=AssumptionType.HARNESS,
        kind=HypothesisKind.VERIFICATION,
        claim=(
            "执行契约假设：当问题要求在预算、时间、发布、迁移或运营风险下给出行动方案时，答案必须把"
            "方法转成最小可逆试点、当前基线、成功指标、停止阈值、责任人和回滚路径；否则即使方法正确，"
            "最终建议也会因不可执行而输给更具体的答案。"
        ),
        context_conditions=[
            "题目要求给出实际行动、排障、迁移、投放或转型方案，而不是只解释理论原理。",
            "存在预算、时间窗口、发布风险、组织能力、停机损失或不可逆投入等现实约束。",
            "答案可以通过小规模试点、灰度、对照、回滚或阶段性验收来降低错误行动成本。",
        ],
        predicted_effects=[
            "把抽象方法落成 judge 可检查的行动契约：做什么、先做到什么程度、何时继续或停止。",
            "减少只有原则没有落地步骤的答案，提升 practical decision / engineering / business rows 的胜率。",
            "给 recursive runner 留下可证伪的 success/stop 条件，方便后续失败归因和 retained hypothesis 学习。",
        ],
        risk_predictions=[
            "纯数学证明、事实 QA 或机制解释题不应强行套执行契约，否则会过度结构化。",
            "如果成功指标与题目目标错位，执行契约会让错误方案显得过度自信。",
        ],
        verifiers=[
            "trigger_subset_pairwise_answer_quality",
            "route_scoped_noop_control_check",
            "success_stop_rollback_fields_present",
        ],
        residual_ids=["res_orthogonal_execution_contract_gap"],
        confidence=0.43,
        metaproductivity=0.07,
        status="candidate",
        tags=[
            "candidate",
            "orthogonal",
            "execution_contract",
            "go_no_go_threshold",
            "rollback_path",
            "decision_owner",
            "answer_harness",
        ],
        source_refs=[
            f"parent:{spec.parent_id}",
            "orthogonal_execution_queue",
            "reconstruction/md/orthogonal_hypothesis_gate_20260608.md",
        ],
        payload={
            "orthogonal_to_existing": True,
            "activation": {
                "problem_ids": spec.trigger_problem_ids,
                "min_keyword_hits": 1,
                "allow_lexical_fallback": False,
            },
            "proposal_type": "orthogonal_failure_hypothesis",
            "orthogonal_axis": spec.axis,
            "residual_cluster": "answer_is_method_correct_but_not_operational",
            "proof_lite_diagram": {
                "objects": ["method_assumption", "task_context", "answer_action", "observed_outcome"],
                "morphisms": [
                    "method_assumption -> answer_action",
                    "task_context -> success_stop_thresholds",
                    "answer_action -> observed_outcome",
                ],
                "invariant": "the answer remains falsifiable and reversible under the task's real constraints",
                "negative_controls": ["pure_proof", "single_fact_lookup", "mechanism_only_explanation"],
            },
            "variation_evaluation_retention": {
                "variation": "new execution-harness axis for residuals where the method is plausible but the answer lacks deployable commitments",
                "evaluation": "expanded trigger/control preflight, live trigger pairwise judge, route-scoped no-op controls",
                "selective_retention": "retain only if trigger benefit clears the candidate acceptance gate without control pollution",
            },
            "scope_repair_note": scope_note,
        },
    )
    edge = AssumptionEdge(
        source=candidate_id,
        target=spec.parent_id,
        type=EdgeType.GENERATED_FROM_RESIDUAL,
        weight=0.53,
        evidence="orthogonal_execution_queue",
        payload={
            "source": "orthogonal_execution_queue",
            "reason": "same residual/parent context, different execution-harness axis",
            "orthogonal_axis": spec.axis,
        },
    )
    return {
        "eval_id": eval_id,
        "source_eval_id": "orthogonal_execution_queue_builder",
        "proposal_counts": {"orthogonal_failure_hypothesis": 1},
        "proposals": [{
            "proposal_id": proposal_id,
            "proposal_type": "orthogonal_failure_hypothesis",
            "parent_node_id": spec.parent_id,
            "candidate_node": candidate.to_dict(),
            "edges": [edge.to_dict()],
            "manifest": None,
            "rationale": (
                "The residual may not be another method-family defect.  It may be that the answer does not "
                "translate the chosen method into a falsifiable, reversible execution contract."
            ),
            "priority": 0.76,
            "source_action": {
                "action_type": "orthogonal_execution_contract",
                "parent_node_id": spec.parent_id,
                "orthogonal_axis": spec.axis,
                "trigger_problem_ids": spec.trigger_problem_ids,
            },
        }],
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
    proposal_id = proposal_payload["proposals"][0]["proposal_id"]
    candidate_id = proposal_payload["proposals"][0]["candidate_node"]["id"]
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        temp_graph = temp_root / "graph"
        _copy_graph(graph_dir, temp_graph)
        before_nodes = set(JsonlGraphStore(temp_graph).nodes)
        judgment_sets = _write_fixture_judgment_sets(
            temp_root=temp_root,
            preflight=preflight,
            proposal_id=proposal_id,
        )
        dry = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight,
            evolution_payload=evolution,
            eval_id=f"{eval_id}_dry",
            queue_name="orthogonal_execution_queue",
            command_limit=1,
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
            queue_name="orthogonal_execution_queue",
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
            queue_name="orthogonal_execution_queue",
            command_limit=1,
            execute=False,
            apply_accepted=True,
            writeback_manifests=True,
        )
        applied_store = JsonlGraphStore(temp_graph)
        temp_orthogonal_edge_count = sum(
            1
            for edge in applied_store.edges
            if edge.source == candidate_id and edge.type == EdgeType.ORTHOGONAL_TO
        )
        candidate_node_present = candidate_id in applied_store.nodes
    return {
        "evolution_payload": evolution,
        "daemon_dry_run": _compact_daemon_payload(dry),
        "fixture_readback": _compact_daemon_payload(readback),
        "fixture_temp_apply": _compact_daemon_payload(applied),
        "node_mutation_without_apply": before_nodes != after_dry_nodes or before_nodes != after_readback_nodes,
        "candidate_node_present_after_apply": candidate_node_present,
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


def _write_fixture_judgment_sets(
    *,
    temp_root: Path,
    preflight: dict[str, Any],
    proposal_id: str,
) -> list[JudgmentSet]:
    baseline_variant = "phase2_v20_gpt54mini_prop_union"
    candidate_variant = f"proposal_{proposal_id.replace('prop_', '')}"
    summary = {row["proposal_id"]: row for row in preflight.get("summaries", [])}[proposal_id]
    rows: dict[str, dict[str, str]] = {}
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
    return [JudgmentSet(
        candidate_variant=candidate_variant,
        baseline_variant=baseline_variant,
        judgment_paths=[path],
        proposal_ids=[proposal_id],
    )]


def _metrics(
    *,
    proposal_payload: dict[str, Any],
    novelty_enabled: dict[str, Any],
    novelty_disabled: dict[str, Any],
    preflight: dict[str, Any],
    readback: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    summary = preflight["summaries"][0] if preflight.get("summaries") else {}
    dry = readback.get("daemon_dry_run", {})
    fixture = readback.get("fixture_readback", {})
    applied = readback.get("fixture_temp_apply", {})
    return {
        "proposal_count": len(proposal_payload.get("proposals", [])),
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
        "preflight_ready_count": preflight.get("readiness_counts", {}).get(
            CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
            0,
        ),
        "trigger_count": len(summary.get("trigger_problem_ids") or []),
        "active_trigger_count": len(summary.get("active_trigger_problem_ids") or []),
        "control_count": len(summary.get("control_problem_ids") or []),
        "outside_active_count": len(summary.get("outside_active_problem_ids") or []),
        "dry_planned_leaf_count": dry.get("planned_leaf_count", 0),
        "dry_executable_leaf_count": dry.get("executable_leaf_count", 0),
        "dry_status_counts": dry.get("execution_status_counts", {}),
        "readback_accept_count": fixture.get("candidate_acceptance_counts", {}).get("accept", 0),
        "readback_resumed": bool(fixture.get("resumed")),
        "readback_applied_count": len(fixture.get("applied_candidate_node_ids") or []),
        "apply_accept_count": applied.get("candidate_acceptance_counts", {}).get("accept", 0),
        "apply_resumed": bool(applied.get("resumed")),
        "apply_applied_count": len(applied.get("applied_candidate_node_ids") or []),
        "node_mutation_without_apply": bool(readback.get("node_mutation_without_apply")),
        "candidate_node_present_after_apply": bool(readback.get("candidate_node_present_after_apply")),
        "temp_orthogonal_edge_count": int(readback.get("temp_orthogonal_edge_count") or 0),
        "live_env_ready": env["gpt"]["ready"] or env["ruoli_gpt"]["ready"],
    }


def _next_commands(
    *,
    root: Path,
    summary: dict[str, Any],
    proposal_id: str,
    proposals_out: Path,
    preflight_out: Path,
) -> list[dict[str, str]]:
    answer_command = (
        "RUOLI_GPT_KEY=<set-in-env> RUOLI_BASE_URL=<set-in-env> GPT_MINI_MODEL=gpt-5.4-mini "
        f"{summary.get('command_hint', '')} --assumption-route-scope-proposals"
    )
    live_command = (
        "RUOLI_GPT_KEY=<set-in-env> RUOLI_BASE_URL=<set-in-env> GPT_MINI_MODEL=gpt-5.4-mini "
        "python3 -m assumption_os.orthogonal_live_ablation --root . "
        f"--queue '{_display_path(root, DEFAULT_OUT)}' "
        "--eval-id orthogonal_execution_live_20260608 --execute-answers --run-judge "
        "--route-scoped-noop-controls "
        "--out 'phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_live_20260608.json'"
    )
    acceptance_command = (
        "python3 -m assumption_os.candidate_acceptance "
        f"--root . --proposals '{_display_path(root, proposals_out)}' "
        f"--preflight '{_display_path(root, preflight_out)}' "
        "--judgments '<judgments-json-from-pairwise-judge>' "
        f"--candidate-variant proposal_{proposal_id.replace('prop_', '')} "
        "--baseline-variant phase2_v20_gpt54mini_prop_union "
        f"--eval-id acceptance_{proposal_id}_orthogonal_execution --proposal-ids {proposal_id} "
        "--summary-out '<acceptance-summary-json>'"
    )
    return [
        {"name": "fresh_ablation_answers", "command": answer_command},
        {"name": "live_route_scoped_ablation", "command": live_command},
        {"name": "acceptance_gate_after_pairwise_judgments", "command": acceptance_command},
    ]


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


def _copy_graph(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


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


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an execution-level orthogonal proposal queue.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--meta", default=str(DEFAULT_META_PATH))
    parser.add_argument("--eval-id", default="orthogonal_execution_queue_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--proposals-out", default=str(DEFAULT_PROPOSALS_OUT))
    parser.add_argument("--preflight-out", default=str(DEFAULT_PREFLIGHT_OUT))
    parser.add_argument(
        "--trigger-problem-ids",
        default="",
        help="comma-separated scoped trigger ids; default uses the broad execution-contract trigger set",
    )
    parser.add_argument("--scope-note", default="")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    proposals_out = _resolve(root, Path(args.proposals_out))
    preflight_out = _resolve(root, Path(args.preflight_out))
    payload = build_orthogonal_execution_queue_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        sample_path=Path(args.sample),
        meta_path=Path(args.meta),
        eval_id=args.eval_id,
        proposals_out=proposals_out,
        preflight_out=preflight_out,
        trigger_problem_ids=_parse_csv(args.trigger_problem_ids) or None,
        scope_note=args.scope_note or None,
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
