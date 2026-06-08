"""Live-ready queue for the strongest orthogonal execution descendant.

The descendant-productivity benchmark shows that keeping the execution-contract
seed as an orthogonal family yields better later descendants.  This module
exports the strongest generation-3 descendant as a normal proposal queue so it
can be tested by the existing live answer/judge pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_eval import build_candidate_eval_payload
from .graph_memory import JsonlGraphStore
from .novelty_integration import build_novelty_integration_payload
from .orthogonal_execution_queue import DEFAULT_META_PATH, DEFAULT_SAMPLE_PATH
from .orthogonal_recursive_ablation import DEFAULT_GRAPH_DIR, PAPER_DIR
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


DEFAULT_OUT = PAPER_DIR / "orthogonal_descendant_live_queue_20260608.json"
DEFAULT_PROPOSALS_OUT = PAPER_DIR / "orthogonal_descendant_live_queue_proposals_20260608.json"
DEFAULT_PREFLIGHT_OUT = PAPER_DIR / "orthogonal_descendant_live_queue_preflight_20260608.json"
DEFAULT_RETAINED_GRAPH_DIR = PAPER_DIR / "orthogonal_descendant_live_graph_20260608"
DEFAULT_SEED_PROPOSALS_PATH = PAPER_DIR / "orthogonal_execution_scope_repair_proposals_20260608.json"

TRIGGER_IDS = [
    "business_0097",
    "business_0192",
    "business_0218",
    "daily_life_0173",
    "software_engineering_0142",
]


def build_orthogonal_descendant_live_queue_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    sample_path: Path | None = None,
    meta_path: Path | None = None,
    eval_id: str | None = None,
    proposals_out: Path | None = None,
    preflight_out: Path | None = None,
    retained_graph_dir: Path | None = None,
    seed_proposals_path: Path | None = None,
) -> dict[str, Any]:
    """Build a live-ready proposal queue for the strongest ON descendant."""

    root = root.resolve()
    base_graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    sample_path = _resolve(root, sample_path or DEFAULT_SAMPLE_PATH)
    meta_path = _resolve(root, meta_path or DEFAULT_META_PATH)
    proposals_out = _resolve(root, proposals_out or DEFAULT_PROPOSALS_OUT)
    preflight_out = _resolve(root, preflight_out or DEFAULT_PREFLIGHT_OUT)
    retained_graph_dir = _resolve(root, retained_graph_dir or DEFAULT_RETAINED_GRAPH_DIR)
    seed_proposals_path = _resolve(root, seed_proposals_path or DEFAULT_SEED_PROPOSALS_PATH)
    eval_id = eval_id or "orthogonal_descendant_live_queue_20260608"

    retained_graph = _prepare_retained_graph_snapshot(
        base_graph_dir=base_graph_dir,
        retained_graph_dir=retained_graph_dir,
        seed_proposals_path=seed_proposals_path,
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
        min_trigger_n=5,
        min_active_trigger_n=5,
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
        env=_env_status(),
    )
    gates = {
        "single_live_descendant": metrics["proposal_count"] == 1,
        "classified_as_specialization_of_execution_family": (
            metrics["specialization_count"] == 1
            and metrics["specializes_edge_count"] == 1
        ),
        "ready_for_fresh_ablation": metrics["preflight_ready_count"] == 1,
        "all_trigger_rows_active": metrics["trigger_count"] >= 5 and metrics["active_trigger_count"] >= 5,
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
        "eval_kind": "orthogonal_descendant_live_ready_queue",
        "performance_validation": True,
        "validation_scope": (
            "Exports the strongest generation-3 orthogonal execution descendant as a live-ready proposal. "
            "This queue validates routing/readback/apply integrity; downstream answer quality is tested by "
            "orthogonal_live_ablation with same-model answers and judge."
        ),
        "status": "live_ready" if metrics["live_env_ready"] else "live_ready_env_missing",
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "base_graph_dir": _display_path(root, base_graph_dir),
            "graph_dir": _display_path(root, retained_graph_dir),
            "retained_graph_dir": _display_path(root, retained_graph_dir),
            "seed_proposals_path": _display_path(root, seed_proposals_path),
            "retained_graph_snapshot": retained_graph,
            "sample_path": _display_path(root, sample_path),
            "meta_path": _display_path(root, meta_path),
            "proposals_out": _display_path(root, proposals_out),
            "preflight_out": _display_path(root, preflight_out),
            "trigger_problem_ids": TRIGGER_IDS,
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
            "This candidate is a descendant, not a new orthogonal seed: it specializes the retained "
            "execution-contract family by combining abstention, compact metric/owner/rollback fields, and "
            "traceable manifests."
        ),
    }


def _build_proposal_payload(*, eval_id: str) -> dict[str, Any]:
    parent_id = "cand_39de0aeae8a3"
    candidate_id = stable_id("cand", eval_id, "constraint_preserving_execution_bridge")
    proposal_id = stable_id("prop", eval_id, candidate_id)
    candidate = AssumptionNode(
        id=candidate_id,
        type=AssumptionType.HARNESS,
        kind=HypothesisKind.VERIFICATION,
        claim=(
            "约束保持执行桥接后代假设：对运营、迁移、投放、转型或通勤优化等行动落地题，先锁定用户"
            "已经给出的硬约束和核心资产，不擅自改目标；再把主体方案桥接到题目特定风险，例如保密边界、"
            "组织能力与角色/KPI、固定目的地与等待方差。最后只追加紧凑执行字段：方案对比、当前基线、"
            "最小可逆试点、责任人与时间窗、go/no-go 阈值、停止或回滚条件。"
        ),
        context_conditions=[
            "问题要求实际行动、迁移、发布、运营、投放、预算、转型或通勤路线决策。",
            "题干含有不可随意放松的约束、核心资产或目标：如专有保密、线下口碑/组织能力、固定训练场地、预算窗口。",
            "答案需要比较选项并保留原约束，再把方法落成可执行、可停止、可回滚的下一步。",
        ],
        predicted_effects=[
            "减少 execution checklist 把用户硬目标改写掉的失败，例如把固定目的地误改成换训练点。",
            "保留上轮 execution-contract 收益，同时补齐 judge 偏好的题目特定桥接：保密分层、方案取舍、角色/KPI 对齐、等待方差控制。",
        ],
        risk_predictions=[
            "如果硬约束并未由题干给出，不应臆造约束；若题目不是行动落地型，应 abstain。",
            "如果清单替代主体推理或写成长模板，会输给更自然且约束更准的 baseline。",
        ],
        verifiers=[
            "same_model_trigger_answer_quality",
            "route_scoped_noop_control_check",
            "hard_constraint_preservation_check",
            "constraint_bridge_field_presence",
        ],
        residual_ids=["res_execution_descendant_live_gap"],
        confidence=0.53,
        metaproductivity=0.14,
        status="candidate",
        tags=[
            "descendant",
            "execution_contract",
            "constraint_bridge",
            "hard_constraint_preservation",
            "abstain_gate",
            "rollback_path",
            "generation_3",
            "live_repair_v2",
        ],
        source_refs=[
            "orthogonal_descendant_productivity_20260608",
            f"parent:{parent_id}",
        ],
        payload={
            "activation": {
                "problem_ids": TRIGGER_IDS,
                "min_keyword_hits": 1,
                "allow_lexical_fallback": False,
            },
            "descendant_generation": 3,
            "parent_family": "orthogonal_execution_contract",
            "variation_evaluation_retention": {
                "variation": "constraint-preserving bridge + compact execution manifest descendant of the accepted execution-contract family",
                "evaluation": "same-model live trigger answers, pairwise judge, route-scoped no-op controls",
                "selective_retention": "retain only if trigger benefit passes without control harm",
            },
        },
    )
    edge = AssumptionEdge(
        source=candidate_id,
        target=parent_id,
        type=EdgeType.SPECIALIZES,
        weight=0.72,
        evidence="orthogonal_descendant_productivity_20260608",
        payload={
            "source": "orthogonal_descendant_live_queue",
            "reason": "best generation-3 descendant of the accepted execution-contract family",
        },
    )
    return {
        "eval_id": eval_id,
        "source_eval_id": "orthogonal_descendant_live_queue_builder",
        "proposal_counts": {"descendant_execution_hypothesis": 1},
        "proposals": [{
            "proposal_id": proposal_id,
            "proposal_type": "descendant_execution_hypothesis",
            "parent_node_id": parent_id,
            "candidate_node": candidate.to_dict(),
            "edges": [edge.to_dict()],
            "manifest": None,
            "rationale": (
                "The first compact-manifest descendant failed live benefit by losing constraint-specific details; "
                "this repair preserves hard constraints before adding execution fields."
            ),
            "priority": 0.84,
            "source_action": {
                "action_type": "export_descendant_for_live_ablation",
                "descendant_generation": 3,
                "repair_variant": "constraint_preserving_execution_bridge_v2",
                "parent_node_id": parent_id,
            },
        }],
    }


def _prepare_retained_graph_snapshot(
    *,
    base_graph_dir: Path,
    retained_graph_dir: Path,
    seed_proposals_path: Path,
) -> dict[str, Any]:
    """Freeze a graph snapshot where the accepted execution seed is retained.

    The main graph is intentionally left untouched.  The descendant is only
    meaningful after the live-positive execution-contract seed has been
    retained, so this snapshot makes that experimental premise explicit.
    """

    _copy_graph(base_graph_dir, retained_graph_dir)
    seed_payload = _load_json(seed_proposals_path)
    proposal = seed_payload["proposals"][0]
    seed_node = AssumptionNode.from_dict(proposal["candidate_node"])
    store = JsonlGraphStore(retained_graph_dir)
    before_node_count = len(store.nodes)
    before_edge_count = len(store.edges)
    store.upsert_node(seed_node)
    for edge in proposal.get("edges", []):
        store.add_edge(AssumptionEdge.from_dict(edge))
    store.add_edge(AssumptionEdge(
        source=seed_node.id,
        target=proposal.get("parent_node_id") or "strategy_S01",
        type=EdgeType.ORTHOGONAL_TO,
        weight=0.42,
        evidence="orthogonal_recursive_ablation_20260608",
        payload={
            "source": "orthogonal_descendant_live_queue",
            "reason": "live-positive execution-contract seed retained as an orthogonal family before descendant testing",
            "proposal_id": proposal.get("proposal_id"),
        },
    ))
    store.flush()
    after = JsonlGraphStore(retained_graph_dir)
    edge_counts = Counter(
        str(edge.type.value if hasattr(edge.type, "value") else edge.type)
        for edge in after.edges
        if edge.source == seed_node.id
    )
    return {
        "path": str(retained_graph_dir),
        "seed_proposal_id": proposal.get("proposal_id"),
        "seed_candidate_node_id": seed_node.id,
        "seed_parent_node_id": proposal.get("parent_node_id"),
        "base_node_count": before_node_count,
        "base_edge_count": before_edge_count,
        "snapshot_node_count": len(after.nodes),
        "snapshot_edge_count": len(after.edges),
        "seed_edge_counts": dict(edge_counts),
        "main_graph_mutated": seed_node.id in JsonlGraphStore(base_graph_dir).nodes,
    }


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
                    "expected_value": 0.61,
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
            queue_name="orthogonal_descendant_live_queue",
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
            queue_name="orthogonal_descendant_live_queue",
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
        "control_count": len(summary.get("control_problem_ids", [])),
        "outside_active_count": len(summary.get("outside_active_problem_ids", [])),
        "readback_accept_count": int(readback_counts.get("accept", 0)),
        "readback_applied_count": len(readback.get("fixture_readback", {}).get("applied_candidate_node_ids", [])),
        "apply_accept_count": int(apply_counts.get("accept", 0)),
        "apply_applied_count": len(readback.get("fixture_temp_apply", {}).get("applied_candidate_node_ids", [])),
        "node_mutation_without_apply": bool(readback.get("node_mutation_without_apply")),
        "candidate_node_present_after_apply": bool(readback.get("candidate_node_present_after_apply")),
        "live_env_ready": bool(env["solver_ready"] and env["judge_ready"]),
    }


def _next_commands(root: Path, proposal_id: str, proposals_out: Path, preflight_out: Path) -> list[dict[str, str]]:
    queue_path = PAPER_DIR / "orthogonal_descendant_live_queue_20260608.json"
    out_path = PAPER_DIR / "orthogonal_descendant_live_same_model_20260608.json"
    return [
        {
            "name": "run_descendant_live_same_model",
            "command": (
                "LLM_PROVIDER=gpt GPT5_API_KEY=<set-in-env> GPT5_BASE_URL=<set-in-env> "
                "GPT5_MODEL=claude-opus-4-8 RUOLI_CLAUDE_KEY=<set-in-env> "
                "CLAUDE_BASE_URL=<set-in-env> CLAUDE_OPUS_MODEL=claude-opus-4-8 "
                "python3 -m assumption_os.orthogonal_live_ablation --root . "
                f"--queue '{_display_path(root, root / queue_path)}' "
                "--eval-id orthogonal_descendant_live_same_model_20260608 "
                "--execute-answers --run-judge --judge-model claude_opus "
                "--baseline-variant phase2_v20_claude_opus_execution_baseline "
                "--route-scoped-noop-controls "
                f"--out '{_display_path(root, root / out_path)}'"
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


def _env_status() -> dict[str, Any]:
    key_ready = bool(os.environ.get("GPT5_API_KEY") or os.environ.get("RUOLI_GPT_KEY") or os.environ.get("RUOLI_CLAUDE_KEY"))
    base_ready = bool(os.environ.get("GPT5_BASE_URL") or os.environ.get("RUOLI_BASE_URL") or os.environ.get("CLAUDE_BASE_URL"))
    return {
        "solver_ready": key_ready and base_ready,
        "judge_ready": key_ready and base_ready,
        "set_names": [
            name
            for name in [
                "GPT5_API_KEY",
                "GPT5_BASE_URL",
                "RUOLI_GPT_KEY",
                "RUOLI_BASE_URL",
                "RUOLI_CLAUDE_KEY",
                "CLAUDE_BASE_URL",
                "CLAUDE_OPUS_MODEL",
            ]
            if os.environ.get(name)
        ],
    }


def _compact_daemon(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ready_queue_count": payload.get("ready_queue_count"),
        "planned_leaf_count": payload.get("planned_leaf_count"),
        "executable_leaf_count": payload.get("executable_leaf_count"),
        "candidate_acceptance_counts": payload.get("candidate_acceptance_counts", {}),
        "accepted_proposal_ids": payload.get("accepted_proposal_ids", []),
        "applied_candidate_node_ids": payload.get("applied_candidate_node_ids", []),
        "resumed": payload.get("resumed"),
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


def _commands_are_secret_free(commands: list[dict[str, str]]) -> bool:
    text = json.dumps(commands, ensure_ascii=False)
    return "sk-" not in text and "newapi_channel_conn" not in text


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
    ap = argparse.ArgumentParser(description="Build live-ready queue for strongest orthogonal descendant.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    ap.add_argument("--meta", default=str(DEFAULT_META_PATH))
    ap.add_argument("--eval-id", default="orthogonal_descendant_live_queue_20260608")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--proposals-out", default=str(DEFAULT_PROPOSALS_OUT))
    ap.add_argument("--preflight-out", default=str(DEFAULT_PREFLIGHT_OUT))
    ap.add_argument("--retained-graph-dir", default=str(DEFAULT_RETAINED_GRAPH_DIR))
    ap.add_argument("--seed-proposals", default=str(DEFAULT_SEED_PROPOSALS_PATH))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_orthogonal_descendant_live_queue_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        sample_path=Path(args.sample),
        meta_path=Path(args.meta),
        eval_id=args.eval_id,
        proposals_out=Path(args.proposals_out),
        preflight_out=Path(args.preflight_out),
        retained_graph_dir=Path(args.retained_graph_dir),
        seed_proposals_path=Path(args.seed_proposals),
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
