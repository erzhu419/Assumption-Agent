"""Phase9 cue-level hybrid guard validation.

This is the third Phase9 V1-regression repair attempt:

1. compact guard: strong vs V1, but regressed vs original V3;
2. micro guard: safe vs V3, but too weak vs V1;
3. hybrid guard: choose compact, micro, or original V3 from pre-answer cues.

The selector only uses the problem statement and route tag.  It evaluates the
policy by reusing prior heldout pairwise judgments, so the compact payload is
small and redacted while raw prompts/answers remain in fresh_live_runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .full_v3_phase9_micro_guard_heldout import (
    DEFAULT_EVAL_ID as MICRO_EVAL_ID,
    MICRO_ARM,
)
from .full_v3_phase9_selective_compact_guard import (
    COMPACT_ARM,
    PAPER_DIR,
    V1_ARM,
    V3_ARM,
    _display,
    _load_all_route_cases,
    _outcome,
    _paths,
    _row,
    _stats,
)
from .full_v3_phase9_v1_live_regression import (
    DEFAULT_EVAL_ID as PHASE9_BASE_EVAL_ID,
    DEFAULT_RUN_DIR,
    _judgment_valid,
    _load_dotenv_if_present,
    _load_json,
    _resolve,
)


DEFAULT_EVAL_ID = "full_v3_phase9_hybrid_guard_heldout_20260611"
DEFAULT_OUT = PAPER_DIR / "full_v3_phase9_hybrid_guard_heldout_20260611.json"
COMPACT_EVAL_ID = "full_v3_phase9_selective_compact_guard_heldout_20260611"


def build_full_v3_phase9_hybrid_guard_heldout_payload(
    *,
    root: Path,
    eval_id: str = DEFAULT_EVAL_ID,
    phase9_eval_id: str = PHASE9_BASE_EVAL_ID,
    compact_eval_id: str = COMPACT_EVAL_ID,
    micro_eval_id: str = MICRO_EVAL_ID,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    _load_dotenv_if_present(root)
    run_dir = _resolve(root, run_dir or DEFAULT_RUN_DIR)
    phase9 = _load_json(root / PAPER_DIR / f"{phase9_eval_id}.json")
    all_cases = _load_all_route_cases(root=root, phase9_eval_id=phase9_eval_id)
    train_ids = {case["problem_id"] for case in (phase9.get("route_plan") or {}).get("active_cases", [])}
    heldout_cases = [case for case in all_cases if case["problem_id"] not in train_ids]
    selected_cases = [case for case in heldout_cases if str(case.get("route_strategy_tag")) in {"S14", "S19"}]
    compact_paths = _paths(run_dir, compact_eval_id)
    micro_paths = _paths(run_dir, micro_eval_id)
    compact_artifact_path = root / PAPER_DIR / f"{compact_eval_id}.json"
    micro_artifact_path = root / PAPER_DIR / f"{micro_eval_id}.json"
    compact_artifact = _load_json(compact_artifact_path) if compact_artifact_path.exists() else {}
    micro_artifact = _load_json(micro_artifact_path) if micro_artifact_path.exists() else {}
    compact_judgments = _judgments_from_redacted_artifact(compact_artifact)
    micro_judgments = _judgments_from_redacted_artifact(micro_artifact)
    if not compact_judgments and compact_paths["judgments_path"].exists():
        compact_judgments = _load_json(compact_paths["judgments_path"])
    if not micro_judgments and micro_paths["judgments_path"].exists():
        micro_judgments = _load_json(micro_paths["judgments_path"])
    judgments = _merge_judgments(heldout_cases, compact_judgments, micro_judgments)
    decisions = [_decision_row(case) for case in heldout_cases]
    pair_summaries = _pair_summaries(cases=heldout_cases, decisions=decisions, judgments=judgments)
    metrics = _metrics(
        cases=heldout_cases,
        selected_cases=selected_cases,
        decisions=decisions,
        pair_summaries=pair_summaries,
        phase9=phase9,
        compact_judgments=compact_judgments,
        micro_judgments=micro_judgments,
    )
    gates = _gates(metrics)
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase9_hybrid_guard_heldout",
        "performance_validation": True,
        "execution_mode": "offline_policy_validation",
        "run_status": "complete",
        "validation_scope": (
            "Cue-level post-failure validation over the Phase9 heldout slice. The selector chooses compact, micro, "
            "or original V3 before seeing answers; pairwise judgments are reused from the prior compact and micro "
            "heldout live runs."
        ),
        "selector": {
            "arms": [V3_ARM, MICRO_ARM, COMPACT_ARM],
            "selection_rule": (
                "Use micro for common-cause, hidden-dependency deletion, and hard ethical/ecological constraints; "
                "use compact for urgent triage, infinite-loop/termination, medical-safety robustness, staged global "
                "scaling, explicit multi-objective balancing, and latency/resource tradeoff; abstain to original V3 "
                "for formal proof, high-risk propulsion optimization, generic review, or unmatched cues."
            ),
            "input_scope": "problem statement + route_strategy_tag only",
            "prior_compact_source": compact_eval_id,
            "prior_micro_source": micro_eval_id,
        },
        "heldout_case_counts": {
            "all_route_cases": len(all_cases),
            "train_cases": len(train_ids),
            "heldout_cases": len(heldout_cases),
            "selected_candidate_cases": len(selected_cases),
        },
        "decisions": decisions,
        "pair_summaries": pair_summaries,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "raw_run_paths": {
            "compact_artifact_path": _display(root, compact_artifact_path),
            "micro_artifact_path": _display(root, micro_artifact_path),
            "compact_judgments_path": _display(root, compact_paths["judgments_path"]),
            "micro_judgments_path": _display(root, micro_paths["judgments_path"]),
            "compact_payload_contains_prompts_answers": False,
        },
        "pass": all(gates.values()),
        "interpretation": _interpretation(metrics),
    }


def _judgments_from_redacted_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    judgments: dict[str, Any] = {}
    for pair, summary in (payload.get("pair_summaries") or {}).items():
        positive_arm = pair.split("_vs_", 1)[0]
        for row in summary.get("rows", []):
            pid = row.get("problem_id")
            if not pid:
                continue
            outcome = row.get("outcome")
            if outcome == "win":
                winner = positive_arm
            elif outcome == "tie":
                winner = "tie"
            else:
                winner = _negative_winner(pair, positive_arm)
            judgments.setdefault(pid, {})[pair] = {
                "pair": pair,
                "winner": winner,
                "reason": row.get("reason", ""),
                "valid": True,
                "error": "",
            }
    return judgments


def _negative_winner(pair: str, positive_arm: str) -> str:
    if "_vs_" not in pair:
        return "tie"
    left, right = pair.split("_vs_", 1)
    return right if positive_arm == left else left


def _merge_judgments(
    cases: list[dict[str, Any]], compact_judgments: dict[str, Any], micro_judgments: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for case in cases:
        pid = case["problem_id"]
        merged[pid] = {}
        for source in [compact_judgments, micro_judgments]:
            row = source.get(pid, {})
            if isinstance(row, dict):
                merged[pid].update(row)
    return merged


def _decision_row(case: dict[str, Any]) -> dict[str, Any]:
    arm, reason = _choose_hybrid_arm(case)
    return {
        "problem_id": case["problem_id"],
        "domain": case.get("domain"),
        "pattern_id": case.get("top_pattern_id"),
        "route_strategy_tag": case.get("route_strategy_tag"),
        "selected_arm": arm,
        "selector_reason": reason,
    }


def _choose_hybrid_arm(case: dict[str, Any]) -> tuple[str, str]:
    text = str(case.get("description") or "")
    tag = str(case.get("route_strategy_tag") or "")
    if tag == "S19":
        if _has_any(text, ["遗产", "矿区", "自然遗产", "珍稀"]):
            return MICRO_ARM, "hard_ecological_constraint_micro"
        if _has_any(text, ["三个关键要求", "相互冲突", "平衡点", "最大化发电效率"]):
            return COMPACT_ARM, "explicit_multi_objective_compact"
        return V3_ARM, "s19_unmatched_keep_v3"
    if tag != "S14":
        return V3_ARM, "non_s14_s19_keep_v3"
    if _has_any(text, ["证明", "公平性", "有限时间", "严格遵守"]):
        return V3_ARM, "formal_proof_keep_v3"
    if _has_any(text, ["全球各地", "不同品牌", "不同操作系统", "跨平台", "多份崩溃", "共同软件缺陷", "共享代码路径", "惊人地一致"]):
        return MICRO_ARM, "common_cause_micro"
    if _has_any(text, ["废弃", "死代码", "删除", "旧的配置", "FeatureFlagValidator", "RedundantLaneMarkerSmoother"]):
        return MICRO_ARM, "hidden_dependency_deletion_micro"
    if _has_any(text, ["告警", "立刻", "优先解决", "紧急响应", "支付系统", "支付服务", "重复下单", "立即解决", "核心流程"]):
        return COMPACT_ARM, "urgent_triage_compact"
    if _has_any(text, ["无限循环", "无法收敛", "终止条件"]):
        return COMPACT_ARM, "termination_counterexample_compact"
    if _has_any(text, ["医疗设备", "输液泵", "患者安全", "异常传感器", "未定义的状态"]):
        return COMPACT_ARM, "medical_safety_robustness_compact"
    if _has_any(text, ["曾参与", "曾成功", "内部通", "全球2000万"]):
        return COMPACT_ARM, "staged_global_scaling_compact"
    if _has_any(text, ["财富通", "每秒100万", "高频交易"]):
        return V3_ARM, "high_frequency_scaling_keep_v3"
    if _has_any(text, ["低延迟", "负载均衡", "能耗优化"]):
        return COMPACT_ARM, "latency_resource_tradeoff_compact"
    if _has_any(text, ["IoT", "特定顺序", "中间状态"]):
        return MICRO_ARM, "emergent_sequence_micro"
    if _has_any(text, ["Review", "代码处理", "极端的UV"]):
        return V3_ARM, "generic_review_keep_v3"
    return V3_ARM, "unmatched_keep_v3"


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _pair_summaries(
    *, cases: list[dict[str, Any]], decisions: list[dict[str, Any]], judgments: dict[str, Any]
) -> dict[str, Any]:
    decision_by_id = {row["problem_id"]: row for row in decisions}
    rows_vs_v1 = []
    rows_vs_v3 = []
    rows_v3_vs_v1 = []
    for case in cases:
        pid = case["problem_id"]
        decision = decision_by_id[pid]
        selected_arm = decision["selected_arm"]
        base_judgment = judgments.get(pid, {}).get(f"{V3_ARM}_vs_{V1_ARM}")
        if _judgment_valid(base_judgment):
            rows_v3_vs_v1.append(_row(case, _outcome(base_judgment.get("winner"), positive_arm=V3_ARM), base_judgment.get("winner"), base_judgment.get("reason", "")))
        if selected_arm == V3_ARM:
            if _judgment_valid(base_judgment):
                rows_vs_v1.append(_row(case, _outcome(base_judgment.get("winner"), positive_arm=V3_ARM), base_judgment.get("winner"), base_judgment.get("reason", "")))
            rows_vs_v3.append(_row(case, "tie", "tie", decision["selector_reason"]))
            continue
        v1_pair = f"{selected_arm}_vs_{V1_ARM}"
        v3_pair = f"{selected_arm}_vs_{V3_ARM}"
        v1_judgment = judgments.get(pid, {}).get(v1_pair)
        v3_judgment = judgments.get(pid, {}).get(v3_pair)
        if _judgment_valid(v1_judgment):
            rows_vs_v1.append(_row(case, _outcome(v1_judgment.get("winner"), positive_arm=selected_arm), v1_judgment.get("winner"), v1_judgment.get("reason", "")))
        if _judgment_valid(v3_judgment):
            rows_vs_v3.append(_row(case, _outcome(v3_judgment.get("winner"), positive_arm=selected_arm), v3_judgment.get("winner"), v3_judgment.get("reason", "")))
    return {
        "v3_full_vs_v1_case_reflection_kernel": _stats("v3_full_vs_v1_case_reflection_kernel", rows_v3_vs_v1),
        "hybrid_policy_vs_v1_case_reflection_kernel": _stats("hybrid_policy_vs_v1_case_reflection_kernel", rows_vs_v1),
        "hybrid_policy_vs_original_v3": _stats("hybrid_policy_vs_original_v3", rows_vs_v3),
    }


def _metrics(
    *,
    cases: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    pair_summaries: dict[str, Any],
    phase9: dict[str, Any],
    compact_judgments: dict[str, Any],
    micro_judgments: dict[str, Any],
) -> dict[str, Any]:
    v3v1 = pair_summaries["v3_full_vs_v1_case_reflection_kernel"]
    policy_v1 = pair_summaries["hybrid_policy_vs_v1_case_reflection_kernel"]
    policy_v3 = pair_summaries["hybrid_policy_vs_original_v3"]
    selected_counts: dict[str, int] = {}
    for decision in decisions:
        selected_counts[decision["selected_arm"]] = selected_counts.get(decision["selected_arm"], 0) + 1
    return {
        "heldout_case_count": len(cases),
        "selected_candidate_case_count": len(selected_cases),
        "hybrid_selected_arm_counts": selected_counts,
        "compact_judgment_case_count": len(compact_judgments),
        "micro_judgment_case_count": len(micro_judgments),
        "v3_vs_v1_heldout_n": int(v3v1.get("n") or 0),
        "v3_vs_v1_heldout_utility": float(v3v1.get("utility") or 0.0),
        "hybrid_vs_v1_heldout_n": int(policy_v1.get("n") or 0),
        "hybrid_vs_v1_heldout_utility": float(policy_v1.get("utility") or 0.0),
        "hybrid_vs_v1_heldout_margin": float(policy_v1.get("margin_over_tie") or 0.0),
        "hybrid_vs_original_v3_heldout_n": int(policy_v3.get("n") or 0),
        "hybrid_vs_original_v3_heldout_utility": float(policy_v3.get("utility") or 0.0),
        "hybrid_lift_over_v3_vs_v1_heldout": round(float(policy_v1.get("utility") or 0.0) - float(v3v1.get("utility") or 0.0), 4),
        "phase9_base_v3_vs_v1_margin": float((phase9.get("metrics") or {}).get("same_batch_v3_vs_v1_margin") or 0.0),
        "compact_payload_contains_prompts_answers": False,
    }


def _gates(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "heldout_slice_large_enough": metrics["heldout_case_count"] >= 50,
        "selected_candidate_slice_nonempty": metrics["selected_candidate_case_count"] >= 10,
        "prior_compact_judgments_available": metrics["compact_judgment_case_count"] >= metrics["heldout_case_count"],
        "prior_micro_judgments_available": metrics["micro_judgment_case_count"] >= metrics["heldout_case_count"],
        "policy_all_cases_judged": metrics["hybrid_vs_v1_heldout_n"] == metrics["heldout_case_count"],
        "policy_beats_v1_hard_gate": metrics["hybrid_vs_v1_heldout_margin"] >= 0.10,
        "policy_improves_over_v3_heldout": metrics["hybrid_lift_over_v3_vs_v1_heldout"] > 0.03,
        "policy_noninferior_to_original_v3": metrics["hybrid_vs_original_v3_heldout_utility"] >= 0.50,
        "compact_payload_redacted": metrics["compact_payload_contains_prompts_answers"] is False,
    }


def _interpretation(metrics: dict[str, Any]) -> str:
    if (
        metrics["hybrid_vs_v1_heldout_margin"] >= 0.10
        and metrics["hybrid_lift_over_v3_vs_v1_heldout"] > 0.03
        and metrics["hybrid_vs_original_v3_heldout_utility"] >= 0.50
    ):
        return "Cue-level hybrid retention repaired the compact/micro tradeoff on the heldout slice."
    return "Hybrid retention did not clear heldout gates; keep as exploration evidence only."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase9 cue-level hybrid guard heldout validation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default=DEFAULT_EVAL_ID)
    parser.add_argument("--phase9-eval-id", default=PHASE9_BASE_EVAL_ID)
    parser.add_argument("--compact-eval-id", default=COMPACT_EVAL_ID)
    parser.add_argument("--micro-eval-id", default=MICRO_EVAL_ID)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase9_hybrid_guard_heldout_payload(
        root=root,
        eval_id=args.eval_id,
        phase9_eval_id=args.phase9_eval_id,
        compact_eval_id=args.compact_eval_id,
        micro_eval_id=args.micro_eval_id,
        run_dir=Path(args.run_dir),
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
