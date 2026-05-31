"""Residual taxonomy helpers.

This module encodes the EmbodiSkill-style distinction that a failed trajectory
does not automatically imply a bad skill/assumption.  The default classifier is
rule-based and deterministic; LLM classifiers can later write into the same
`ResidualAssessment` shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ResidualType, TrialManifest


@dataclass(frozen=True)
class ResidualAssessment:
    residual_type: ResidualType
    reason: str
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "residual_type": self.residual_type.value,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
        }


def classify_manifest(manifest: TrialManifest) -> ResidualAssessment:
    text = " ".join(
        str(x or "")
        for x in (
            manifest.residual,
            manifest.observed_effect,
            manifest.expected_effect,
            manifest.why_selected,
        )
    ).lower()

    if not manifest.residual:
        return ResidualAssessment(
            ResidualType.NO_RESIDUAL,
            "No residual was recorded.",
            "Reinforce or leave confidence unchanged.",
        )
    if _has_any(text, ["not applied", "didn't apply", "did not apply", "没用", "未应用", "没真正执行", "空转", "装饰"]):
        return ResidualAssessment(
            ResidualType.EXECUTION_LAPSE,
            "The active assumption appears valid but was not executed faithfully.",
            "Preserve the assumption; add execution reminder, constraint, or verifier.",
        )
    if _has_any(text, ["partly", "partial", "方向对", "不够具体", "不够强", "refine", "优化"]):
        return ResidualAssessment(
            ResidualType.OPTIMIZATION,
            "The assumption direction helped but needs tighter operating conditions or payload.",
            "Refine signal/unpacked/formal payload and retest against outside controls.",
        )
    if _has_any(text, ["judge", "verifier", "判官", "评分", "verbosity", "风格", "generic warning"]):
        return ResidualAssessment(
            ResidualType.EVALUATOR_DEFECT,
            "The verifier may be measuring the wrong target or over-weighting style.",
            "Audit with objective gold, placebo/generic control, and cross-family judges.",
        )
    if _has_any(text, ["retrieval", "memory", "检索", "记忆", "wrong memory", "irrelevant"]):
        return ResidualAssessment(
            ResidualType.MEMORY_DEFECT,
            "The retrieval/memory policy likely activated the wrong context.",
            "Update graph edges, recognition filter, or router before changing the assumption.",
        )
    if _has_any(text, ["world model", "simulator", "预测错", "rollout", "brier", "auroc"]):
        return ResidualAssessment(
            ResidualType.SIMULATOR_DEFECT,
            "The cheap predictor/world model appears miscalibrated.",
            "Collect more simulator labels; do not trust this rollout for library edits.",
        )
    if _has_any(text, ["wrong", "false", "contradict", "过度泛化", "条件错", "不成立", "defect"]):
        return ResidualAssessment(
            ResidualType.ASSUMPTION_DEFECT,
            "The assumption itself appears false or overgeneralized under the current conditions.",
            "Narrow applicability, add anti-assumption, or deprecate after fresh validation.",
        )
    return ResidualAssessment(
        ResidualType.DISCOVERY,
        "The residual is not explained by current assumption categories.",
        "Cluster with similar residuals and generate a new candidate only if systematic.",
    )


def _has_any(text: str, needles: list[str]) -> bool:
    return any(n.lower() in text for n in needles)
