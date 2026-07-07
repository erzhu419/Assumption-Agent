"""Live queue for a technical-execution descendant of the accepted family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .candidate_eval import build_candidate_eval_payload
from .graph_memory import JsonlGraphStore
from .novelty_integration import build_novelty_integration_payload
from .orthogonal_descendant_live_queue import (
    DEFAULT_OUT as DEFAULT_PARENT_QUEUE,
    DEFAULT_RETAINED_GRAPH_DIR as DEFAULT_SOURCE_GRAPH_DIR,
    _commands_are_secret_free,
    _display_path,
    _env_status,
    _load_json,
    _resolve,
)
from .orthogonal_descendant_nextgen_live_queue import (
    PARENT_CANDIDATE_ID,
    SEED_CANDIDATE_ID,
    _evolution_payload,
    _fixture_readback,
    _prepare_nextgen_graph_snapshot,
)
from .orthogonal_execution_queue import DEFAULT_META_PATH, DEFAULT_SAMPLE_PATH
from .orthogonal_recursive_ablation import PAPER_DIR
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    stable_id,
)


DEFAULT_OUT = PAPER_DIR / "orthogonal_technical_descendant_live_queue_20260609.json"
DEFAULT_PROPOSALS_OUT = PAPER_DIR / "orthogonal_technical_descendant_live_queue_proposals_20260609.json"
DEFAULT_PREFLIGHT_OUT = PAPER_DIR / "orthogonal_technical_descendant_live_queue_preflight_20260609.json"
DEFAULT_RETAINED_GRAPH_DIR = PAPER_DIR / "orthogonal_technical_descendant_live_graph_20260609"
DEFAULT_BASELINE_VARIANT = "phase2_v20_claude_opus_technical_baseline"

TECHNICAL_TRIGGER_IDS = [
    "engineering_0244",
    "software_engineering_0142",
    "software_engineering_0379",
]
TECHNICAL_ABSTAINED_IDS = [
    "software_engineering_0265",
    "software_engineering_0337",
]


def build_orthogonal_technical_descendant_live_queue_payload(
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
    baseline_variant: str = DEFAULT_BASELINE_VARIANT,
) -> dict[str, Any]:
    root = root.resolve()
    source_graph_dir = _resolve(root, source_graph_dir or DEFAULT_SOURCE_GRAPH_DIR)
    parent_queue_path = _resolve(root, parent_queue_path or DEFAULT_PARENT_QUEUE)
    sample_path = _resolve(root, sample_path or DEFAULT_SAMPLE_PATH)
    meta_path = _resolve(root, meta_path or DEFAULT_META_PATH)
    proposals_out = _resolve(root, proposals_out or DEFAULT_PROPOSALS_OUT)
    preflight_out = _resolve(root, preflight_out or DEFAULT_PREFLIGHT_OUT)
    retained_graph_dir = _resolve(root, retained_graph_dir or DEFAULT_RETAINED_GRAPH_DIR)
    eval_id = eval_id or "orthogonal_technical_descendant_live_queue_20260609"

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
        min_trigger_n=len(TECHNICAL_TRIGGER_IDS),
        min_active_trigger_n=len(TECHNICAL_TRIGGER_IDS),
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
        "single_technical_descendant": metrics["proposal_count"] == 1,
        "accepted_parent_retained": (
            retained_graph["seed_candidate_node_id"] == SEED_CANDIDATE_ID
            and retained_graph["parent_candidate_node_id"] == PARENT_CANDIDATE_ID
            and retained_graph["parent_status_after_snapshot"] == "active"
        ),
        "classified_as_specialization": (
            metrics["specialization_count"] == 1
            and metrics["specializes_edge_count"] == 1
        ),
        "ready_for_fresh_ablation": metrics["preflight_ready_count"] == 1,
        "all_technical_triggers_active": (
            metrics["trigger_count"] == len(TECHNICAL_TRIGGER_IDS)
            and metrics["active_trigger_count"] == len(TECHNICAL_TRIGGER_IDS)
            and set(metrics["trigger_problem_ids"]) == set(TECHNICAL_TRIGGER_IDS)
        ),
        "failed_or_tie_residuals_abstained": not (set(TECHNICAL_ABSTAINED_IDS) & set(metrics["trigger_problem_ids"])),
        "controls_present": metrics["control_count"] >= 8,
        "no_outside_activation": metrics["outside_active_count"] == 0,
        "fixture_readback_accepts": metrics["readback_accept_count"] == 1,
        "readback_without_apply_does_not_mutate_graph": (
            metrics["readback_applied_count"] == 0
            and not metrics["node_mutation_without_apply"]
        ),
        "fixture_temp_apply_writes_candidate": metrics["apply_applied_count"] == 1,
        "commands_are_secret_free": _commands_are_secret_free(_next_commands(root, proposal_id, proposals_out, preflight_out, baseline_variant)),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_technical_descendant_live_ready_queue",
        "performance_validation": True,
        "validation_scope": (
            "Exports a technical/engineering execution descendant generated from the GraphQL migration residual. "
            "This repaired version applies selective retention after a 5-trigger live run: it keeps the three "
            "technical rows that the candidate won and abstains on the API-DX loss and release-pipeline tie."
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
            "trigger_problem_ids": TECHNICAL_TRIGGER_IDS,
            "abstained_residual_problem_ids": TECHNICAL_ABSTAINED_IDS,
            "baseline_variant": baseline_variant,
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
        "next_commands": _next_commands(root, proposal_id, proposals_out, preflight_out, baseline_variant),
        "interpretation": (
            "This branch tests whether the negative software/API migration residual can produce a productive "
            "technical-execution child instead of being forced into the business-oriented descendant.  The current "
            "scope is selectively retained from a rejected 5-trigger technical run."
        ),
    }


def _build_proposal_payload(*, eval_id: str) -> dict[str, Any]:
    candidate_id = stable_id("cand", eval_id, "technical_execution_risk_governance_bridge_retained_wins_repair")
    proposal_id = stable_id("prop", eval_id, candidate_id)
    candidate = AssumptionNode(
        id=candidate_id,
        type=AssumptionType.HARNESS,
        kind=HypothesisKind.VERIFICATION,
        claim=(
            "技术执行风险治理保留假设：当软件/工程题同时有黑盒机制、资源/时间约束、上线或融资压力、"
            "以及潜在回归风险，并且任务本质是迁移、调参或研发方向止损时，答案不能只给技术方案，而要"
            "先把问题重写成可证伪的风险治理实验。"
            "通用结构是：建立 P95/P99 或业务 KPI 基线，找关键路径/瓶颈或心智模型错位；先用低成本治理"
            "拿确定收益，再做局部试点；主动寻找击穿反例；设置 SLA、统计显著性或 stop-loss 阈值；所有改动"
            "必须能灰度、dual-run、kill-switch 或回滚。具体映射：黑盒语音延迟用插桩+DOE/贝叶斯优化；"
            "再用极值粗筛冻结不敏感参数，剩余 2-3 个参数才做小交互测试，并设置诊断预算、P99 接受线、"
            "超时降级/回滚兜底；GraphQL 迁移先承认 REST 锁定资产和融资窗口，低成本加固 OpenAPI+运行时校验，"
            "再用 BFF 试点、dual-run、kill-switch 红队 N+1/鉴权/缓存/攻击面，并用前后端/投资人可见指标对齐；"
            "风控模型边际收益递减时用在线配对实验、标签/数据信噪比诊断、不可证伪信念识别、canary 和研发止损点。"
            "对 API DX 摩擦和发布管线题先 abstain，"
            "等待单独的 DX-prototype 或 pipeline-bottleneck 子假设。"
        ),
        formal_form={
            "category_inspired_diagram": {
                "objects": [
                    "observed_technical_residual",
                    "risk_governed_experiment",
                    "selectively_retained_technical_scope",
                    "bounded_rollout_decision",
                ],
                "morphisms": [
                    {
                        "source": "observed_technical_residual",
                        "target": "risk_governed_experiment",
                        "preserves": ["delivery_pressure", "technical_uncertainty"],
                        "adds": ["baseline_metric", "falsification_probe", "counterexample_search"],
                    },
                    {
                        "source": "risk_governed_experiment",
                        "target": "selectively_retained_technical_scope",
                        "preserves": ["baseline_metric", "counterexample_search"],
                        "adds": ["abstain_on_dx_loss", "abstain_on_pipeline_tie"],
                    },
                    {
                        "source": "selectively_retained_technical_scope",
                        "target": "bounded_rollout_decision",
                        "preserves": ["baseline_metric", "counterexample_search", "technical_scope_boundary"],
                        "adds": ["dual_run", "kill_switch", "stop_loss_threshold"],
                    },
                ],
                "invariants": [
                    "measure_before_migrating_or_tuning",
                    "prefer_reversible_local_trial_before_full_rewrite",
                    "search_for_failure_modes_before_scale_up",
                    "tie_action_to_sla_or_business_kpi",
                    "freeze_insensitive_variables_before_interaction_tests",
                ],
                "certificate_type": "bounded_structural_morphism_proof_lite",
            }
        },
        context_conditions=[
            "问题属于软件工程、工程调试、模型研发决策或技术迁移。",
            "题干包含时间/资源/融资/上线压力，并且存在全量重写、盲目调参或模型复杂化的诱惑。",
            "答案需要给出可量化基线、局部试点、反例搜索、回滚/熔断和 go/no-go 阈值。",
            "当前保留作用域不包括 API DX 摩擦诊断和发布管线优化；这两类需单独子假设验证。",
        ],
        predicted_effects=[
            "把技术题从方案罗列改成风险受控的实验和发布决策，提升 judge 偏好的约束处理和可执行性。",
            "避免 full rewrite、all-in tuning 或追求复杂模型导致的不可逆回归。",
            "把 GraphQL 负例里的 path-dependency、attack surface、stakeholder alignment 推广成技术执行范畴。",
            "补齐 retained child 上一轮输掉的极值粗筛、冻结无关参数、超时降级和融资窗口利益相关方对齐。",
            "通过 selective retention 移除 API DX 和发布管线上的未验证泛化，降低 benefit 回归。",
        ],
        risk_predictions=[
            "如果题目是纯数学证明或无上线风险的开放解释题，应 abstain。",
            "如果答案只机械套用 dual-run/kill-switch 而不连接具体技术机制，会输给更自然的 baseline。",
        ],
        verifiers=[
            "same_model_technical_trigger_quality",
            "route_scoped_noop_control_check",
            "baseline_metric_presence",
            "counterexample_and_stoploss_presence",
        ],
        residual_ids=["res_software_migration_scope_loss_20260608"],
        confidence=0.54,
        metaproductivity=0.2,
        status="candidate",
        tags=[
            "descendant",
            "technical_execution",
            "risk_governance",
            "counterexample_search",
            "dual_run",
            "kill_switch",
            "bounded_rollout",
            "selective_retention",
            "extreme_value_screening",
            "stakeholder_alignment",
            "proof_lite_diagram",
            "generation_4",
        ],
        source_refs=[
            "orthogonal_descendant_live_same_model_20260608",
            "software_engineering_0142_live_loss_residual",
            f"parent:{PARENT_CANDIDATE_ID}",
        ],
        payload={
            "activation": {
                "problem_ids": TECHNICAL_TRIGGER_IDS,
                "min_keyword_hits": 1,
                "allow_lexical_fallback": False,
            },
            "abstained_residual_problem_ids": TECHNICAL_ABSTAINED_IDS,
            "descendant_generation": 4,
            "parent_family": "orthogonal_execution_contract",
            "parent_candidate_node_id": PARENT_CANDIDATE_ID,
            "variation_evaluation_retention": {
                "variation": "residual-specific repair of the selectively retained technical-execution child",
                "evaluation": "same-model baseline/candidate answers, pairwise judge, route-scoped no-op controls",
                "selective_retention": "retain the three live-winning technical triggers; abstain on API-DX loss and pipeline tie",
                "repair_from_last_judgment": {
                    "engineering_0244": "add extreme-value coarse screening, freeze insensitive params before interaction tests, and timeout-degradation fallback",
                    "software_engineering_0142": "make path dependency, security attack surface, stakeholder alignment, and kill-switch thresholds explicit",
                },
            },
        },
    )
    edge = AssumptionEdge(
        source=candidate_id,
        target=PARENT_CANDIDATE_ID,
        type=EdgeType.SPECIALIZES,
        weight=0.74,
        evidence="software_engineering_0142_live_loss_residual",
        payload={
            "source": "orthogonal_technical_descendant_live_queue",
            "reason": "sibling technical child generated from the accepted descendant's software/API migration loss",
        },
    )
    return {
        "eval_id": eval_id,
        "source_eval_id": "orthogonal_technical_descendant_live_queue_builder",
        "proposal_counts": {"descendant_execution_hypothesis": 1},
        "proposals": [{
            "proposal_id": proposal_id,
            "proposal_type": "descendant_execution_hypothesis",
            "parent_node_id": PARENT_CANDIDATE_ID,
            "candidate_node": candidate.to_dict(),
            "edges": [edge.to_dict()],
            "manifest": None,
            "rationale": (
                "The accepted business/action descendant lost on GraphQL migration because it lacked technical "
                "risk governance details.  This branch tests that residual as a technical-execution child."
            ),
            "priority": 0.83,
            "source_action": {
                "action_type": "export_technical_descendant_for_live_ablation",
                "descendant_generation": 4,
                "parent_node_id": PARENT_CANDIDATE_ID,
                "abstained_residual_problem_ids": TECHNICAL_ABSTAINED_IDS,
            },
        }],
    }


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
        "parent_snapshot_specializes_count": int(retained_graph["parent_edge_counts"].get(EdgeType.SPECIALIZES.value, 0)),
        "readback_accept_count": int(readback_counts.get("accept", 0)),
        "readback_applied_count": len(readback.get("fixture_readback", {}).get("applied_candidate_node_ids", [])),
        "apply_accept_count": int(apply_counts.get("accept", 0)),
        "apply_applied_count": len(readback.get("fixture_temp_apply", {}).get("applied_candidate_node_ids", [])),
        "node_mutation_without_apply": bool(readback.get("node_mutation_without_apply")),
        "candidate_node_present_after_apply": bool(readback.get("candidate_node_present_after_apply")),
        "live_env_ready": bool(env["solver_ready"] and env["judge_ready"]),
    }


def _next_commands(
    root: Path,
    proposal_id: str,
    proposals_out: Path,
    preflight_out: Path,
    baseline_variant: str,
) -> list[dict[str, str]]:
    queue_path = DEFAULT_OUT
    out_path = PAPER_DIR / "orthogonal_technical_descendant_live_same_model_20260609.json"
    baseline_sample = "phase two/analysis/cache/proposal_samples/orthogonal_technical_descendant_retained_baseline_20260609_sample.json"
    return [
        {
            "name": "generate_same_model_technical_baseline",
            "command": (
                "LLM_PROVIDER=gpt GPT5_API_KEY=<set-in-env> GPT5_BASE_URL=<set-in-env> "
                "GPT5_MODEL=claude-opus-4-8 python3 'phase one/scripts/validation/phase2_v20_framework.py' "
                f"--variant {baseline_variant} --sample '{baseline_sample}' --n {len(TECHNICAL_TRIGGER_IDS)}"
            ),
        },
        {
            "name": "run_technical_descendant_live_same_model",
            "command": (
                "LLM_PROVIDER=gpt GPT5_API_KEY=<set-in-env> GPT5_BASE_URL=<set-in-env> "
                "GPT5_MODEL=claude-opus-4-8 RUOLI_CLAUDE_KEY=<set-in-env> "
                "CLAUDE_BASE_URL=<set-in-env> CLAUDE_OPUS_MODEL=claude-opus-4-8 "
                "python3 -m assumption_os.orthogonal_live_ablation --root . "
                f"--queue '{_display_path(root, root / queue_path)}' "
                "--eval-id orthogonal_technical_descendant_live_same_model_20260609 "
                "--execute-answers --run-judge --judge-model claude_opus "
                f"--baseline-variant {baseline_variant} "
                "--route-scoped-noop-controls "
                f"--out '{_display_path(root, root / out_path)}'"
            ),
        },
        {"name": "proposal_payload", "command": _display_path(root, proposals_out)},
        {"name": "preflight_payload", "command": _display_path(root, preflight_out)},
        {"name": "proposal_id", "command": proposal_id},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build live queue for a technical orthogonal descendant.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--source-graph-dir", default=str(DEFAULT_SOURCE_GRAPH_DIR))
    ap.add_argument("--parent-queue", default=str(DEFAULT_PARENT_QUEUE))
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    ap.add_argument("--meta", default=str(DEFAULT_META_PATH))
    ap.add_argument("--eval-id", default="orthogonal_technical_descendant_live_queue_20260609")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--proposals-out", default=str(DEFAULT_PROPOSALS_OUT))
    ap.add_argument("--preflight-out", default=str(DEFAULT_PREFLIGHT_OUT))
    ap.add_argument("--retained-graph-dir", default=str(DEFAULT_RETAINED_GRAPH_DIR))
    ap.add_argument("--baseline-variant", default=DEFAULT_BASELINE_VARIANT)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    payload = build_orthogonal_technical_descendant_live_queue_payload(
        root=root,
        source_graph_dir=Path(args.source_graph_dir),
        parent_queue_path=Path(args.parent_queue),
        sample_path=Path(args.sample),
        meta_path=Path(args.meta),
        eval_id=args.eval_id,
        proposals_out=Path(args.proposals_out),
        preflight_out=Path(args.preflight_out),
        retained_graph_dir=Path(args.retained_graph_dir),
        baseline_variant=args.baseline_variant,
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
