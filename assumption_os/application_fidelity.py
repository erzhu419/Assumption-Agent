"""Audit whether answer-time OperatorSpecs were actually applied.

This is a cheap verifier for the execution layer.  It does not try to prove
answer quality; it checks whether required operator slots are materially present
in the final answer so retrieved assumptions are less likely to remain
decorative context.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .operator_specs import OperatorSpec


SLOT_CUES: dict[str, list[str]] = {
    "variable_or_cause_changed": ["改变", "调整", "变量", "因素", "原因", "干预", "change", "changed", "variable", "cause", "factor", "intervention"],
    "variables_held_constant": ["保持", "固定", "不变", "控制", "其他条件", "同一", "held constant", "fixed", "constant"],
    "control_or_baseline": ["对照", "基线", "baseline", "control", "control group", "a/b", "ab test"],
    "observed_metric": ["指标", "衡量", "观察", "测量", "数据", "转化", "成本", "率", "metric", "measure", "outcome", "kpi"],
    "decision_rule": ["如果", "则", "阈值", "规则", "标准", "决策", "加码", "停止", "止损", "decision", "threshold", "rule", "if "],
    "preserved_behavior": ["保留", "保持", "兼容", "现有", "行为", "接口", "流程", "preserve", "existing behavior", "workflow"],
    "incremental_slice": ["小步", "切片", "阶段", "增量", "最小", "试点", "slice", "increment", "pilot", "phase"],
    "adapter_or_boundary": ["适配", "边界", "接口", "封装", "adapter", "boundary", "interface", "wrapper"],
    "acceptance_metric": ["验收", "指标", "通过", "检查", "sla", "acceptance", "metric", "pass"],
    "rollback_path": ["回滚", "回退", "恢复", "兜底", "fallback", "rollback", "restore"],
    "source_roles": ["源", "原", "角色", "对应", "source", "role"],
    "target_roles": ["目标", "新问题", "角色", "映射", "target", "role"],
    "preserved_invariant": ["不变量", "保持", "结构", "机制", "invariant", "preserve"],
    "limiting_case": ["极限", "退化", "边界情况", "limiting case", "degenerate", "boundary case"],
    "negative_control": ["负对照", "失败场景", "不适用", "negative control", "should fail"],
    "decisive_evidence": ["关键证据", "直接证据", "decisive", "evidence", "source"],
    "entity_scope_boundary": ["范围", "边界", "对象", "实体", "scope", "entity", "boundary"],
    "answer_bearing_relation": ["关系", "支持结论", "直接回答", "relation", "answer-bearing"],
    "overturn_condition": ["推翻", "反驳", "改变结论", "overturn", "falsify"],
    "trigger": ["触发", "问题在于", "因为", "trigger"],
    "applied_constraint": ["约束", "必须", "因此", "constraint"],
    "evidence_or_boundary": ["证据", "边界", "范围", "evidence", "boundary"],
    "verifier_or_failure_condition": ["验证", "失败条件", "检查", "verifier", "failure condition"],
}

TERM_CUES: dict[str, list[str]] = {
    "variable": ["变量", "因素"],
    "cause": ["原因", "因果"],
    "changed": ["改变", "调整"],
    "held": ["保持", "固定"],
    "constant": ["不变", "固定"],
    "control": ["对照", "控制"],
    "baseline": ["基线"],
    "metric": ["指标", "衡量"],
    "decision": ["决策", "判断"],
    "rule": ["规则", "标准"],
    "preserved": ["保留", "保持"],
    "behavior": ["行为", "流程"],
    "incremental": ["增量", "分阶段"],
    "slice": ["切片", "小步"],
    "adapter": ["适配器", "适配"],
    "boundary": ["边界"],
    "rollback": ["回滚", "回退"],
    "source": ["源", "原"],
    "target": ["目标"],
    "roles": ["角色"],
    "invariant": ["不变量", "结构"],
    "limiting": ["极限", "退化"],
    "negative": ["负对照", "不适用"],
    "evidence": ["证据"],
    "scope": ["范围"],
    "verifier": ["验证器", "检查"],
}


@dataclass(frozen=True)
class SlotCheck:
    slot: str
    present: bool
    cues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorApplicationAudit:
    source_id: str
    used: bool
    decorative: bool
    misapplied: bool
    slot_completion_rate: float
    required_slots: list[str]
    filled_slots: list[str]
    missing_slots: list[str]
    slot_checks: list[SlotCheck]
    negative_control_hits: list[str]
    fidelity_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slot_checks"] = [check.to_dict() for check in self.slot_checks]
        return payload


def audit_operator_application(
    answer: str,
    spec: OperatorSpec | dict[str, Any],
    *,
    min_slot_completion: float = 0.65,
) -> OperatorApplicationAudit:
    operator = coerce_operator_spec(spec)
    checks = [
        SlotCheck(slot=slot, present=_slot_present(answer, slot), cues=_slot_cues(slot))
        for slot in operator.required_output_slots
    ]
    filled = [check.slot for check in checks if check.present]
    missing = [check.slot for check in checks if not check.present]
    slot_rate = len(filled) / len(checks) if checks else 1.0
    negative_hits = _negative_control_hits(answer, operator)
    operator_mentioned = _operator_mentioned(answer, operator)
    decorative = bool(operator_mentioned and slot_rate < min_slot_completion)
    misapplied = bool(negative_hits and slot_rate < min_slot_completion)
    used = bool(slot_rate >= min_slot_completion and not misapplied)
    fidelity = _bounded(slot_rate - (0.25 if decorative else 0.0) - (0.35 if misapplied else 0.0))
    return OperatorApplicationAudit(
        source_id=operator.source_id,
        used=used,
        decorative=decorative,
        misapplied=misapplied,
        slot_completion_rate=round(slot_rate, 4),
        required_slots=list(operator.required_output_slots),
        filled_slots=filled,
        missing_slots=missing,
        slot_checks=checks,
        negative_control_hits=negative_hits,
        fidelity_score=round(fidelity, 4),
    )


def audit_answer_application(
    answer: str,
    specs: Iterable[OperatorSpec | dict[str, Any]],
    *,
    min_slot_completion: float = 0.65,
) -> dict[str, Any]:
    audits = [
        audit_operator_application(answer, spec, min_slot_completion=min_slot_completion)
        for spec in specs
    ]
    if not audits:
        return {
            "operator_count": 0,
            "used_assumption_ids": [],
            "ignored_assumption_ids": [],
            "misapplied_assumption_ids": [],
            "decorative_use_count": 0,
            "slot_completion_rate": 1.0,
            "application_fidelity": 1.0,
            "pass": True,
            "operators": [],
        }
    used = [audit.source_id for audit in audits if audit.used]
    misapplied = [audit.source_id for audit in audits if audit.misapplied]
    ignored = [audit.source_id for audit in audits if not audit.used and not audit.decorative and not audit.misapplied]
    decorative_count = sum(1 for audit in audits if audit.decorative)
    slot_rate = sum(audit.slot_completion_rate for audit in audits) / len(audits)
    fidelity = sum(audit.fidelity_score for audit in audits) / len(audits)
    return {
        "operator_count": len(audits),
        "used_assumption_ids": used,
        "ignored_assumption_ids": ignored,
        "misapplied_assumption_ids": misapplied,
        "decorative_use_count": decorative_count,
        "slot_completion_rate": round(slot_rate, 4),
        "application_fidelity": round(fidelity, 4),
        "pass": bool(fidelity >= min_slot_completion and decorative_count == 0 and not misapplied),
        "operators": [audit.to_dict() for audit in audits],
    }


def coerce_operator_spec(spec: OperatorSpec | dict[str, Any]) -> OperatorSpec:
    if isinstance(spec, OperatorSpec):
        return spec
    payload = dict(spec)
    payload.setdefault("source_claim", "")
    payload.setdefault("trigger_conditions", [])
    payload.setdefault("execution_steps", [])
    payload.setdefault("required_output_slots", [])
    payload.setdefault("negative_controls", [])
    payload.setdefault("verifier_checks", [])
    payload.setdefault("fallback_policy", "")
    payload.setdefault("confidence", 0.0)
    return OperatorSpec(**payload)


def _slot_present(answer: str, slot: str) -> bool:
    text = _norm(answer)
    return any(_norm(cue) in text for cue in _slot_cues(slot))


def _slot_cues(slot: str) -> list[str]:
    cues = list(SLOT_CUES.get(slot, []))
    for term in re.split(r"[_\W]+", slot.lower()):
        if not term:
            continue
        cues.append(term)
        cues.extend(TERM_CUES.get(term, []))
    return _dedupe([cue for cue in cues if cue])


def _negative_control_hits(answer: str, spec: OperatorSpec) -> list[str]:
    text = _norm(answer)
    hits: list[str] = []
    for control in spec.negative_controls:
        low = _norm(control)
        if "changing many" in low or "many causal factors" in low:
            if _contains_any(text, ["同时改变", "全部改变", "一次性改变", "change all", "change many"]) and not _contains_any(text, ["对照", "固定", "baseline", "control"]):
                hits.append(control)
        elif "full rewrite" in low or "full replacement" in low:
            if _contains_any(text, ["推倒重来", "全量重写", "full rewrite", "replace all"]) and not _contains_any(text, ["回滚", "rollback", "兼容", "preserve"]):
                hits.append(control)
    return hits


def _claim_mentioned(answer: str, claim: str) -> bool:
    claim_terms = [tok for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|[\u4e00-\u9fff]{2,}", claim.lower()) if len(tok) >= 2]
    if not claim_terms:
        return False
    text = _norm(answer)
    meaningful = [term for term in claim_terms if term not in {"use", "the", "and", "for", "before", "after"}]
    if not meaningful:
        return False
    return sum(1 for term in meaningful if _norm(term) in text) >= min(2, len(meaningful))


def _operator_mentioned(answer: str, spec: OperatorSpec) -> bool:
    if _claim_mentioned(answer, spec.source_claim):
        return True
    text = _norm(answer)
    if _contains_any(text, ["控制变量", "controlled variable", "control variable"]):
        return True
    if _contains_any(text, ["增量替换", "增量迁移", "strangler", "adapter", "适配器"]):
        return True
    if _contains_any(text, ["类比", "结构迁移", "不变量", "limiting case", "negative control"]):
        return True
    slot_cue_hits = 0
    for slot in spec.required_output_slots:
        if any(_norm(cue) in text for cue in _slot_cues(slot)[:4]):
            slot_cue_hits += 1
    return slot_cue_hits >= 2


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower())


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(_norm(needle) in text for needle in needles)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))
