"""Fast-policy memory primitives for continual Assumption-Agent learning.

The HLE source/operator branch should not promote one-off seed patches into the
slow baseline.  This module keeps learned experience in a small fast layer:
typed policy hypotheses, deterministic trigger/anti-trigger scoring, and a
cost/stability-aware promotion gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable

from .autonomy_journal import stable_hash


FAST_POLICY_MEMORY_VERSION = "fast_policy_memory_v1"

FAST_POLICY_KINDS = {
    "operator",
    "source_binding",
    "solver_lane",
    "fallback_policy",
}

PROMOTION_STATUSES = {
    "candidate",
    "shadow",
    "promoted",
    "regression_only",
    "rejected",
    "deprecated",
}

DEFAULT_TRACKED_FAILURE_BUCKETS = (
    "candidate_generation_missed_gold",
    "source_verifier_no_candidate_emitted",
    "no_selected_label_generic",
    "source_quality_directness_promotion_blocked",
    "verified_or_abstain_no_fallback",
    "verified_or_abstain no_fallback",
)


@dataclass(frozen=True)
class FastPolicyHypothesis:
    """A reusable fast-weight policy, not a committed slow-baseline change."""

    id: str
    kind: str
    action: str
    trigger_terms: list[str]
    anti_trigger_terms: list[str] = field(default_factory=list)
    expected_utility: float = 0.0
    expected_harm: float = 0.0
    evidence_rows: list[dict[str, Any]] = field(default_factory=list)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)
    promotion_status: str = "candidate"
    fallback_behavior: str = "preserve_slow_baseline"
    source_refs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.id:
            issues.append("id_missing")
        if self.kind not in FAST_POLICY_KINDS:
            issues.append("invalid_kind")
        if not self.action:
            issues.append("action_missing")
        if not _normalize_terms(self.trigger_terms):
            issues.append("trigger_terms_missing")
        if self.expected_utility < 0.0 or self.expected_utility > 1.0:
            issues.append("expected_utility_out_of_range")
        if self.expected_harm < 0.0 or self.expected_harm > 1.0:
            issues.append("expected_harm_out_of_range")
        if self.promotion_status not in PROMOTION_STATUSES:
            issues.append("invalid_promotion_status")
        if not self.fallback_behavior:
            issues.append("fallback_behavior_missing")
        return issues

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_version"] = FAST_POLICY_MEMORY_VERSION
        payload["validation_issues"] = self.validate()
        payload["raw_content_persisted"] = False
        return payload


@dataclass(frozen=True)
class PolicyScoreRow:
    policy_id: str
    kind: str
    action: str
    promotion_status: str
    trigger_hit_count: int
    anti_trigger_hit_count: int
    p_trigger: float
    p_harm: float
    selected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionGateSpec:
    """Cost/stability-aware gate for promoting a fast policy."""

    fixed_regression_min_accuracy: float = 0.5
    min_unseen_correct_gain: int = 2
    selected_label_stability_min: float = 0.95
    max_no_fallback_count: int = 0
    max_cost_ratio: float = 1.2
    allow_noninferior_with_stability: bool = True
    tracked_failure_buckets: tuple[str, ...] = DEFAULT_TRACKED_FAILURE_BUCKETS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_fast_policy(
    policy: FastPolicyHypothesis | dict[str, Any],
    *,
    problem_text: str = "",
    features: dict[str, Any] | None = None,
    allowed_statuses: set[str] | None = None,
) -> PolicyScoreRow:
    """Score trigger fit without retaining raw question content."""

    hyp = _coerce_policy(policy)
    allowed = allowed_statuses or {"promoted"}
    if hyp.validate():
        return PolicyScoreRow(
            policy_id=hyp.id,
            kind=hyp.kind,
            action=hyp.action,
            promotion_status=hyp.promotion_status,
            trigger_hit_count=0,
            anti_trigger_hit_count=0,
            p_trigger=0.0,
            p_harm=1.0,
            selected=False,
            reason="invalid_policy",
        )
    tokens = _problem_tokens(problem_text=problem_text, features=features)
    trigger_terms = set(_normalize_terms(hyp.trigger_terms))
    anti_terms = set(_normalize_terms(hyp.anti_trigger_terms))
    trigger_hits = len(tokens & trigger_terms)
    anti_hits = len(tokens & anti_terms)
    trigger_rate = trigger_hits / max(1, len(trigger_terms))
    anti_rate = anti_hits / max(1, len(anti_terms)) if anti_terms else 0.0
    p_trigger = min(1.0, trigger_rate + 0.25 * hyp.expected_utility)
    p_harm = min(1.0, anti_rate + 0.35 * hyp.expected_harm)
    selected = (
        hyp.promotion_status in allowed
        and trigger_hits > 0
        and p_trigger > p_harm
        and hyp.expected_utility >= hyp.expected_harm
    )
    if hyp.promotion_status not in allowed:
        reason = "status_not_allowed_for_decision"
    elif trigger_hits == 0:
        reason = "trigger_miss"
    elif p_trigger <= p_harm:
        reason = "anti_trigger_or_harm_blocks"
    elif hyp.expected_utility < hyp.expected_harm:
        reason = "expected_harm_exceeds_utility"
    else:
        reason = "selected"
    return PolicyScoreRow(
        policy_id=hyp.id,
        kind=hyp.kind,
        action=hyp.action,
        promotion_status=hyp.promotion_status,
        trigger_hit_count=trigger_hits,
        anti_trigger_hit_count=anti_hits,
        p_trigger=round(p_trigger, 4),
        p_harm=round(p_harm, 4),
        selected=selected,
        reason=reason,
    )


def select_fast_policies(
    policies: Iterable[FastPolicyHypothesis | dict[str, Any]],
    *,
    problem_text: str = "",
    features: dict[str, Any] | None = None,
    allowed_statuses: set[str] | None = None,
    max_selected: int = 3,
) -> dict[str, Any]:
    """Select promoted fast policies while preserving slow-baseline fallback."""

    rows = [
        score_fast_policy(
            policy,
            problem_text=problem_text,
            features=features,
            allowed_statuses=allowed_statuses,
        )
        for policy in policies
    ]
    rows = sorted(rows, key=lambda row: (row.selected, row.p_trigger - row.p_harm, row.p_trigger), reverse=True)
    selected_rows = [row for row in rows if row.selected][: max(0, max_selected)]
    selected_ids = [row.policy_id for row in selected_rows]
    selected_kinds = sorted({row.kind for row in selected_rows})
    payload = {
        "policy_version": FAST_POLICY_MEMORY_VERSION,
        "selected_policy_ids": selected_ids,
        "selected_policy_kinds": selected_kinds,
        "selected_actions": [row.action for row in selected_rows],
        "slow_baseline_required": True,
        "fallback_behavior": "preserve_slow_baseline",
        "score_rows": [row.to_dict() for row in rows],
        "question_hash": stable_hash({"problem_text": problem_text}) if problem_text else None,
        "feature_hash": stable_hash(features or {}),
        "raw_content_persisted": False,
    }
    payload["fast_policy_payload_hash"] = stable_hash({
        "policy_version": FAST_POLICY_MEMORY_VERSION,
        "selected_policy_ids": selected_ids,
        "selected_actions": payload["selected_actions"],
        "feature_hash": payload["feature_hash"],
    })
    return payload


def evaluate_fast_policy_promotion(
    *,
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    fixed_regression_metrics: dict[str, Any] | None = None,
    gate: PromotionGateSpec | None = None,
) -> dict[str, Any]:
    """Evaluate whether a fast policy can be promoted beyond shadow use."""

    spec = gate or PromotionGateSpec()
    fixed = fixed_regression_metrics or candidate_metrics
    candidate_unseen_correct = _metric_int(candidate_metrics, "unseen_correct", "correct")
    baseline_unseen_correct = _metric_int(baseline_metrics, "unseen_correct", "correct")
    candidate_unseen_total = max(1, _metric_int(candidate_metrics, "unseen_total", "total"))
    baseline_unseen_total = max(1, _metric_int(baseline_metrics, "unseen_total", "total"))
    candidate_unseen_accuracy = candidate_unseen_correct / candidate_unseen_total
    baseline_unseen_accuracy = baseline_unseen_correct / baseline_unseen_total
    fixed_accuracy = _metric_float(fixed, "fixed_regression_accuracy", "accuracy")
    if fixed_accuracy is None:
        fixed_correct = _metric_int(fixed, "fixed_regression_correct", "correct")
        fixed_total = max(1, _metric_int(fixed, "fixed_regression_total", "total"))
        fixed_accuracy = fixed_correct / fixed_total
    no_fallback_count = _metric_int(candidate_metrics, "no_fallback_count")
    stability = _metric_float(candidate_metrics, "selected_label_stability") or 0.0
    candidate_cost = _metric_float(candidate_metrics, "cost", "unique_model_calls")
    baseline_cost = _metric_float(baseline_metrics, "cost", "unique_model_calls")
    clear_accuracy_gain = candidate_unseen_correct >= baseline_unseen_correct + spec.min_unseen_correct_gain
    noninferior = candidate_unseen_accuracy >= baseline_unseen_accuracy
    cost_ok = True
    cost_ratio = None
    if candidate_cost is not None and baseline_cost and baseline_cost > 0:
        cost_ratio = candidate_cost / baseline_cost
        cost_ok = cost_ratio <= spec.max_cost_ratio or clear_accuracy_gain
    bucket_delta = _failure_bucket_delta(candidate_metrics, baseline_metrics, spec.tracked_failure_buckets)
    bucket_nonworse = all(value <= 0 for value in bucket_delta.values())
    stability_gain_ok = (
        spec.allow_noninferior_with_stability
        and noninferior
        and stability >= spec.selected_label_stability_min
        and no_fallback_count <= spec.max_no_fallback_count
        and cost_ok
    )
    gates = {
        "fixed_regression_noninferior": fixed_accuracy >= spec.fixed_regression_min_accuracy,
        "unseen_accuracy_gain_or_stable_noninferior": clear_accuracy_gain or stability_gain_ok,
        "selected_label_stability": stability >= spec.selected_label_stability_min,
        "no_fallback_count_zero": no_fallback_count <= spec.max_no_fallback_count,
        "cost_within_budget_or_accuracy_gain": cost_ok,
        "tracked_failure_buckets_nonworse": bucket_nonworse,
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    decision = "promote_fast_policy" if not failed_gates else "shadow_only"
    return {
        "policy_version": FAST_POLICY_MEMORY_VERSION,
        "promotion_allowed": not failed_gates,
        "decision": decision,
        "failed_gates": failed_gates,
        "gates": gates,
        "gate_spec": spec.to_dict(),
        "metrics": {
            "fixed_regression_accuracy": round(fixed_accuracy, 4),
            "candidate_unseen_accuracy": round(candidate_unseen_accuracy, 4),
            "baseline_unseen_accuracy": round(baseline_unseen_accuracy, 4),
            "unseen_correct_gain": candidate_unseen_correct - baseline_unseen_correct,
            "selected_label_stability": stability,
            "no_fallback_count": no_fallback_count,
            "cost_ratio": round(cost_ratio, 4) if cost_ratio is not None else None,
            "failure_bucket_delta": bucket_delta,
        },
        "raw_content_persisted": False,
    }


def _coerce_policy(policy: FastPolicyHypothesis | dict[str, Any]) -> FastPolicyHypothesis:
    if isinstance(policy, FastPolicyHypothesis):
        return policy
    return FastPolicyHypothesis(
        id=str(policy.get("id", "")),
        kind=str(policy.get("kind", "")),
        action=str(policy.get("action", "")),
        trigger_terms=list(policy.get("trigger_terms", []) or []),
        anti_trigger_terms=list(policy.get("anti_trigger_terms", []) or []),
        expected_utility=float(policy.get("expected_utility", 0.0) or 0.0),
        expected_harm=float(policy.get("expected_harm", 0.0) or 0.0),
        evidence_rows=list(policy.get("evidence_rows", []) or []),
        failure_rows=list(policy.get("failure_rows", []) or []),
        promotion_status=str(policy.get("promotion_status", "candidate")),
        fallback_behavior=str(policy.get("fallback_behavior", "preserve_slow_baseline")),
        source_refs=list(policy.get("source_refs", []) or []),
        notes=list(policy.get("notes", []) or []),
    )


def _normalize_terms(values: Iterable[Any]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip().lower()
        if not item:
            continue
        for token in re.findall(r"[a-z0-9_+-]+", item):
            if token and token not in seen:
                seen.add(token)
                terms.append(token)
    return terms


def _problem_tokens(*, problem_text: str, features: dict[str, Any] | None) -> set[str]:
    chunks = [problem_text]
    for key, value in (features or {}).items():
        chunks.append(str(key))
        if isinstance(value, (list, tuple, set)):
            chunks.extend(str(item) for item in value)
        elif isinstance(value, dict):
            chunks.extend(str(k) for k in value)
            chunks.extend(str(v) for v in value.values())
        else:
            chunks.append(str(value))
    return set(_normalize_terms(chunks))


def _metric_int(metrics: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if value is not None and str(value).strip():
            try:
                return int(float(str(value)))
            except ValueError:
                continue
    return 0


def _metric_float(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if value is not None and str(value).strip():
            try:
                return float(str(value))
            except ValueError:
                continue
    return None


def _failure_bucket_delta(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    tracked: Iterable[str],
) -> dict[str, int]:
    candidate = Counter(candidate_metrics.get("failure_buckets") or {})
    baseline = Counter(baseline_metrics.get("failure_buckets") or {})
    return {bucket: int(candidate.get(bucket, 0) - baseline.get(bucket, 0)) for bucket in tracked}
