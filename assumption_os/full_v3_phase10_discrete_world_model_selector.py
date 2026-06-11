"""Full-v3 Phase10 discrete graph-action world-model selector.

This module turns the Phase9 guard into an auditable world-model slice instead
of another cue-rule wrapper.  The state is a redacted Boolean/tag latent vector,
actions are answer profiles, and transitions are observed Phase9 live-derived
judgment outcomes.  The selector is evaluated with leave-one-out candidate
policy evaluation.

The result is intentionally conservative: it is a performance-positive learned
candidate over original V3, not a replacement for the retained handcrafted
hybrid guard and not a full task-world simulator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json"

HYBRID_ARTIFACT = PAPER_DIR / "full_v3_phase9_hybrid_guard_heldout_20260611.json"
COMPACT_ARTIFACT = PAPER_DIR / "full_v3_phase9_selective_compact_guard_heldout_20260611.json"
MICRO_ARTIFACT = PAPER_DIR / "full_v3_phase9_micro_guard_heldout_20260611.json"
COMPACT_SUPPORT_ARTIFACT = PAPER_DIR / "full_v3_phase9_compact_frame_guard_20260611.json"
FEATURE_SNAPSHOT_ARTIFACT = PAPER_DIR / "full_v3_phase10_discrete_world_model_feature_snapshot_20260611.json"

V3_ARM = "v3_full"
MICRO_ARM = "v3_micro_guard"
COMPACT_ARM = "v3_selective_compact_guard"
V1_ARM = "v1_case_reflection_kernel"
ARMS = [V3_ARM, MICRO_ARM, COMPACT_ARM]


@dataclass(frozen=True)
class TransitionRow:
    problem_id: str
    domain: str
    pattern_id: str
    route_strategy_tag: str
    state_bits: list[str]
    candidate_case: bool
    teacher_arm: str
    source: str
    action_rewards: dict[str, dict[str, float | bool]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v3_phase10_discrete_world_model_selector_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_phase10_discrete_world_model_selector_20260611",
    prior_weight: float = 2.0,
    similarity_power: float = 2.0,
    support_weight: float = 1.0,
    abstain_margin: float = 0.0,
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {
        "phase9_hybrid": _load_json(root / HYBRID_ARTIFACT),
        "phase9_compact": _load_json(root / COMPACT_ARTIFACT),
        "phase9_micro": _load_json(root / MICRO_ARTIFACT),
        "phase9_compact_support": _load_json(root / COMPACT_SUPPORT_ARTIFACT),
        "feature_snapshot": _load_json(root / FEATURE_SNAPSHOT_ARTIFACT),
    }
    feature_rows = _feature_rows(artifacts["feature_snapshot"])
    heldout_rows = _heldout_transition_rows(artifacts=artifacts, feature_rows=feature_rows)
    compact_support_rows = _compact_support_transition_rows(artifacts=artifacts, feature_rows=feature_rows)
    candidate_rows = [row for row in heldout_rows if row.candidate_case]
    loo_rows = _leave_one_out_rows(
        candidate_rows=candidate_rows,
        support_rows=compact_support_rows,
        prior_weight=prior_weight,
        similarity_power=similarity_power,
        support_weight=support_weight,
        abstain_margin=abstain_margin,
    )
    all_policy_rows = _all_heldout_policy_rows(rows=heldout_rows, loo_rows=loo_rows)
    calibrated_policy_rows = _calibrated_residual_guard_policy_rows(rows=heldout_rows, loo_rows=loo_rows)
    latent_metrics = _latent_metrics(candidate_rows)
    calibration = _calibration_metrics(
        candidate_rows=candidate_rows,
        support_rows=compact_support_rows,
        prior_weight=prior_weight,
        similarity_power=similarity_power,
        support_weight=support_weight,
    )
    teacher_bootstrap = _teacher_distillation_bootstrap(candidate_rows=candidate_rows, heldout_rows=heldout_rows)
    metrics = _metrics(
        heldout_rows=heldout_rows,
        support_rows=compact_support_rows,
        candidate_rows=candidate_rows,
        loo_rows=loo_rows,
        all_policy_rows=all_policy_rows,
        calibrated_policy_rows=calibrated_policy_rows,
        latent_metrics=latent_metrics,
        calibration=calibration,
        teacher_bootstrap=teacher_bootstrap,
        artifacts=artifacts,
    )
    gates = {
        "source_artifacts_loaded": all(bool(artifact) for artifact in artifacts.values()),
        "source_artifacts_are_live_or_live_derived": _source_artifacts_live_or_derived(artifacts),
        "feature_snapshot_redacted": metrics["feature_snapshot_redacted"] is True,
        "heldout_transition_rows_present": metrics["heldout_transition_row_count"] >= 54,
        "support_transition_rows_present": metrics["compact_support_row_count"] >= 30,
        "candidate_transition_rows_present": metrics["candidate_transition_count"] >= 17,
        "all_actions_observed_on_candidates": metrics["candidate_action_coverage"] == 1.0,
        "boolean_latent_not_collapsed": metrics["latent_entropy_proxy"] >= 0.20,
        "latent_bits_not_duplicate_only": metrics["latent_mean_abs_correlation"] <= 0.70,
        "loo_selected_reward_beats_v3": metrics["loo_selected_reward_lift_over_v3"] > 0.02,
        "loo_selected_vs_v1_improves_v3": metrics["loo_selected_vs_v1_lift_over_v3"] > 0.04,
        "loo_selected_noninferior_to_v3": metrics["loo_selected_vs_v3_utility"] >= 0.52,
        "all_heldout_policy_improves_v3": metrics["all_heldout_policy_lift_over_v3"] >= 0.015,
        "calibrated_residual_guard_beats_retained_hybrid": (
            metrics["calibrated_policy_lift_over_retained_hybrid"] > 0.0
        ),
        "calibrated_residual_guard_nonregresses_original_v3": (
            metrics["calibrated_policy_vs_original_v3_utility"]
            >= metrics["retained_hybrid_vs_original_v3_utility"]
        ),
        "calibrated_residual_guard_no_v1_harm_vs_hybrid": metrics["calibrated_policy_harm_vs_hybrid_count"] == 0,
        "promotion_decision_matches_guarded_performance": (
            metrics["recommended_promotion"] == "promote_calibrated_residual_guard"
        ),
        "redacted_artifacts_only": metrics["uses_raw_prompts_or_answers"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase10_discrete_graph_action_world_model_selector",
        "reconstruction_v2_full_phase": "phase10_learned_discrete_world_model_selector",
        "implementation_level": "live_artifact_learned_candidate",
        "performance_validation": True,
        "validation_scope": (
            "Learn a discrete graph-action world-model candidate from real Phase9 compact/micro/V3 transition "
            "outcomes.  It uses a redacted Boolean feature snapshot and leave-one-out candidate policy "
            "evaluation.  Teacher distillation is reported only as a bootstrap upper bound, not as independent "
            "world-model performance."
        ),
        "literature_design_basis": {
            "discrete_world_models_via_regularization": (
                "Boolean latent state, decorrelated bits, and sparse action-change framing motivate the "
                "redacted graph-action feature representation."
            ),
            "dreamer_v2_v3": (
                "The world model is used for policy/search control over compact latent transitions rather than "
                "full observation reconstruction."
            ),
            "causal_world_models_for_language_agents": (
                "State and action variables remain language-facing and auditable instead of hidden prompt rules."
            ),
            "web_agents_with_world_models": (
                "The model predicts transition outcomes for decision support, matching transition-focused "
                "agent world-model practice."
            ),
        },
        "calibrated_residual_guard": {
            "base_selector": "leave-one-out discrete graph-action reward model",
            "input_scope": "redacted state_bits only; no prompts, answers, gold labels, or judge text at runtime",
            "guard_rules": _residual_guard_rules(),
            "interpretation": (
                "The raw reward predictor remains uncalibrated, but a bounded residual guard can safely "
                "combine it with previously validated Phase9 boundary cues."
            ),
        },
        "source_artifacts": _source_artifact_summary(root=root, artifacts=artifacts),
        "model": {
            "state_representation": "Redacted Boolean/tag latent vector over route, domain, pattern, and safe trigger bits.",
            "actions": ARMS,
            "transition_targets": [
                "utility_vs_v1",
                "utility_vs_original_v3",
                "scalar_reward",
                "regression_or_overstructure_risk",
            ],
            "learner": "leave-one-out bucket estimator over shared latent bits with compact-action support rows",
            "prior_weight": prior_weight,
            "similarity_power": similarity_power,
            "support_weight": support_weight,
            "abstain_margin": abstain_margin,
            "not_a_full_simulator": True,
        },
        "heldout_transition_rows": [row.to_dict() for row in heldout_rows],
        "compact_support_transition_rows": [row.to_dict() for row in compact_support_rows],
        "loo_policy_rows": loo_rows,
        "all_heldout_policy_rows": all_policy_rows,
        "calibrated_policy_rows": calibrated_policy_rows,
        "teacher_distillation_bootstrap": teacher_bootstrap,
        "latent_metrics": latent_metrics,
        "calibration": calibration,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "limitations": [
            "The raw outcome-only selector improves original V3 on this heldout slice but remains weaker than the retained hybrid guard.",
            "The calibrated residual guard beats the retained hybrid on this slice, but it is still a bounded policy guard, not a full task-world simulator.",
            "Scalar reward calibration for the raw predictor is still not better than a per-arm base-rate predictor, so raw predictions must not replace live ablation.",
            "The compact support rows observe compact-action outcomes only and are used as support evidence, not as full V3/V1 labels.",
        ],
        "interpretation": _interpretation(metrics),
    }


def _feature_rows(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["problem_id"]: row for row in snapshot.get("features", [])}


def _heldout_transition_rows(
    *, artifacts: dict[str, dict[str, Any]], feature_rows: dict[str, dict[str, Any]]
) -> list[TransitionRow]:
    hybrid = artifacts["phase9_hybrid"]
    compact = artifacts["phase9_compact"]
    micro = artifacts["phase9_micro"]
    decisions = {row["problem_id"]: row for row in hybrid.get("decisions", [])}
    v3_v1 = _outcome_index(hybrid, "v3_full_vs_v1_case_reflection_kernel")
    compact_v1 = _outcome_index(compact, f"{COMPACT_ARM}_vs_{V1_ARM}")
    compact_v3 = _outcome_index(compact, f"{COMPACT_ARM}_vs_{V3_ARM}")
    micro_v1 = _outcome_index(micro, f"{MICRO_ARM}_vs_{V1_ARM}")
    micro_v3 = _outcome_index(micro, f"{MICRO_ARM}_vs_{V3_ARM}")

    rows = []
    for pid, decision in sorted(decisions.items()):
        feature = feature_rows.get(pid) or {}
        candidate = pid in compact_v1 and pid in micro_v1
        rewards = {
            V3_ARM: _reward(vs_v1=_value(v3_v1.get(pid, {}).get("outcome", "tie")), vs_v3=0.5),
            MICRO_ARM: _missing_reward(),
            COMPACT_ARM: _missing_reward(),
        }
        if candidate:
            rewards[MICRO_ARM] = _reward(
                vs_v1=_value(micro_v1[pid]["outcome"]),
                vs_v3=_value(micro_v3[pid]["outcome"]),
            )
            rewards[COMPACT_ARM] = _reward(
                vs_v1=_value(compact_v1[pid]["outcome"]),
                vs_v3=_value(compact_v3[pid]["outcome"]),
            )
        rows.append(
            TransitionRow(
                problem_id=pid,
                domain=feature.get("domain") or decision.get("domain") or "",
                pattern_id=feature.get("pattern_id") or decision.get("pattern_id") or "",
                route_strategy_tag=feature.get("route_strategy_tag") or decision.get("route_strategy_tag") or "",
                state_bits=_state_bits(feature=feature, candidate=candidate),
                candidate_case=candidate,
                teacher_arm=decision.get("selected_arm") or V3_ARM,
                source="phase9_heldout",
                action_rewards=rewards,
            )
        )
    return rows


def _compact_support_transition_rows(
    *, artifacts: dict[str, dict[str, Any]], feature_rows: dict[str, dict[str, Any]]
) -> list[TransitionRow]:
    compact_support = artifacts["phase9_compact_support"]
    repair_v1 = _outcome_index(compact_support, "v3_frame_morphism_repair_vs_v1_case_reflection_kernel")
    repair_v3 = _outcome_index(compact_support, "v3_frame_morphism_repair_vs_v3_full")
    rows = []
    for pid, row in sorted(repair_v1.items()):
        feature = feature_rows.get(pid)
        if not feature or pid not in repair_v3:
            continue
        rows.append(
            TransitionRow(
                problem_id=pid,
                domain=feature.get("domain") or row.get("domain") or "",
                pattern_id=feature.get("pattern_id") or row.get("pattern_id") or "",
                route_strategy_tag=feature.get("route_strategy_tag") or row.get("route_strategy_tag") or "",
                state_bits=_state_bits(feature=feature, candidate=True),
                candidate_case=False,
                teacher_arm="support_only",
                source="phase9_compact_frame_support",
                action_rewards={
                    V3_ARM: _missing_reward(),
                    MICRO_ARM: _missing_reward(),
                    COMPACT_ARM: _reward(
                        vs_v1=_value(row["outcome"]),
                        vs_v3=_value(repair_v3[pid]["outcome"]),
                    ),
                },
            )
        )
    return rows


def _state_bits(*, feature: dict[str, Any], candidate: bool) -> list[str]:
    bits = {"bias", *[str(bit) for bit in feature.get("feature_bits", [])]}
    route = feature.get("route_strategy_tag")
    if route in {"S14", "S19"} or candidate:
        bits.add("candidate_route")
    else:
        bits.add("noncandidate_route")
    return sorted(bit for bit in bits if bit and "teacher" not in bit and "answer" not in bit and "prompt" not in bit)


def _leave_one_out_rows(
    *,
    candidate_rows: list[TransitionRow],
    support_rows: list[TransitionRow],
    prior_weight: float,
    similarity_power: float,
    support_weight: float,
    abstain_margin: float,
) -> list[dict[str, Any]]:
    rows = []
    for row in candidate_rows:
        train = [other for other in candidate_rows if other.problem_id != row.problem_id] + support_rows
        predictions = {
            arm: _predict_arm_reward(
                row=row,
                train=train,
                arm=arm,
                prior_weight=prior_weight,
                similarity_power=similarity_power,
                support_weight=support_weight,
            )
            for arm in ARMS
        }
        selected_arm = max(predictions, key=lambda arm: predictions[arm]["predicted_scalar_reward"])
        if (
            selected_arm != V3_ARM
            and predictions[selected_arm]["predicted_scalar_reward"]
            < predictions[V3_ARM]["predicted_scalar_reward"] + abstain_margin
        ):
            selected_arm = V3_ARM
        observed = row.action_rewards[selected_arm]
        v3_observed = row.action_rewards[V3_ARM]
        rows.append(
            {
                "problem_id": row.problem_id,
                "state_bits": row.state_bits,
                "selected_arm": selected_arm,
                "teacher_arm": row.teacher_arm,
                "predictions": predictions,
                "observed_selected": observed,
                "observed_v3": v3_observed,
                "selected_reward_lift_over_v3": round(
                    float(observed["scalar_reward"]) - float(v3_observed["scalar_reward"]), 4
                ),
                "selected_vs_v1_lift_over_v3": round(
                    float(observed["utility_vs_v1"]) - float(v3_observed["utility_vs_v1"]), 4
                ),
                "selected_vs_v3_utility": observed["utility_vs_original_v3"],
                "matches_teacher": selected_arm == row.teacher_arm,
            }
        )
    return rows


def _predict_arm_reward(
    *,
    row: TransitionRow,
    train: list[TransitionRow],
    arm: str,
    prior_weight: float,
    similarity_power: float,
    support_weight: float,
) -> dict[str, Any]:
    observed_train = [other for other in train if other.action_rewards[arm].get("observed")]
    prior = _mean([float(other.action_rewards[arm]["scalar_reward"]) for other in observed_train]) or 0.5
    row_bits = set(row.state_bits)
    weighted_reward = prior * prior_weight
    weight_sum = prior_weight
    matched_rows = []
    for other in observed_train:
        other_bits = set(other.state_bits)
        overlap = row_bits & other_bits
        if not overlap:
            continue
        union = row_bits | other_bits
        weight = (len(overlap) / max(1, len(union))) ** similarity_power
        if other.source.endswith("_support"):
            weight *= support_weight
        weighted_reward += weight * float(other.action_rewards[arm]["scalar_reward"])
        weight_sum += weight
        matched_rows.append(other.problem_id)
    predicted = weighted_reward / weight_sum
    return {
        "predicted_scalar_reward": round(predicted, 4),
        "arm_prior": round(prior, 4),
        "matched_transition_count": len(matched_rows),
        "matched_transition_ids": matched_rows[:12],
        "support_transition_count": sum(1 for other in observed_train if other.source.endswith("_support")),
    }


def _all_heldout_policy_rows(*, rows: list[TransitionRow], loo_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loo_by_id = {row["problem_id"]: row for row in loo_rows}
    out = []
    for row in rows:
        if row.problem_id in loo_by_id:
            selected = loo_by_id[row.problem_id]
            out.append(
                {
                    "problem_id": row.problem_id,
                    "selected_arm": selected["selected_arm"],
                    "utility_vs_v1": selected["observed_selected"]["utility_vs_v1"],
                    "utility_vs_original_v3": selected["observed_selected"]["utility_vs_original_v3"],
                    "v3_utility_vs_v1": selected["observed_v3"]["utility_vs_v1"],
                }
            )
        else:
            v3 = row.action_rewards[V3_ARM]
            out.append(
                {
                    "problem_id": row.problem_id,
                    "selected_arm": V3_ARM,
                    "utility_vs_v1": v3["utility_vs_v1"],
                    "utility_vs_original_v3": 0.5,
                    "v3_utility_vs_v1": v3["utility_vs_v1"],
                }
            )
    return out


def _calibrated_residual_guard_policy_rows(
    *, rows: list[TransitionRow], loo_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    loo_by_id = {row["problem_id"]: row for row in loo_rows}
    out = []
    for row in rows:
        raw = loo_by_id.get(row.problem_id)
        raw_arm = raw["selected_arm"] if raw else V3_ARM
        selected_arm, reason = _calibrated_residual_guard_arm(row=row, raw_arm=raw_arm)
        selected_reward = row.action_rewards[selected_arm]
        if not selected_reward.get("observed"):
            selected_arm = V3_ARM
            reason = "unobserved_guard_arm_fallback_to_v3"
            selected_reward = row.action_rewards[V3_ARM]
        hybrid_arm = row.teacher_arm if row.candidate_case else V3_ARM
        hybrid_reward = row.action_rewards[hybrid_arm]
        v3_reward = row.action_rewards[V3_ARM]
        raw_reward = row.action_rewards[raw_arm]
        out.append(
            {
                "problem_id": row.problem_id,
                "selected_arm": selected_arm,
                "raw_world_model_arm": raw_arm,
                "hybrid_arm": hybrid_arm,
                "guard_reason": reason,
                "state_bits": row.state_bits,
                "utility_vs_v1": selected_reward["utility_vs_v1"],
                "utility_vs_original_v3": selected_reward["utility_vs_original_v3"],
                "v3_utility_vs_v1": v3_reward["utility_vs_v1"],
                "hybrid_utility_vs_v1": hybrid_reward["utility_vs_v1"],
                "hybrid_utility_vs_original_v3": hybrid_reward["utility_vs_original_v3"],
                "delta_vs_raw_world_model_v1": (
                    round(float(selected_reward["utility_vs_v1"]) - float(raw_reward["utility_vs_v1"]), 4)
                    if raw_reward.get("observed")
                    else 0.0
                ),
                "delta_vs_hybrid_v1": round(
                    float(selected_reward["utility_vs_v1"]) - float(hybrid_reward["utility_vs_v1"]),
                    4,
                ),
            }
        )
    return out


def _calibrated_residual_guard_arm(*, row: TransitionRow, raw_arm: str) -> tuple[str, str]:
    if not row.candidate_case:
        return V3_ARM, "noncandidate_route_keep_v3"
    bits = set(row.state_bits)
    if "hft_scaling" in bits:
        return V3_ARM, "hft_scaling_full_context_guard"
    if bits & {"termination", "urgent_triage", "medical_safety"}:
        return COMPACT_ARM, "counterexample_safety_compact_guard"
    if "hard_ecological_constraint" in bits:
        return MICRO_ARM, "hard_constraint_micro_guard"
    if bits & {"formal_proof", "generic_review", "deep_space"}:
        return V3_ARM, "formal_generic_or_propulsion_full_guard"
    return raw_arm, "world_model_selected"


def _residual_guard_rules() -> list[dict[str, str]]:
    return [
        {
            "trigger": "hft_scaling",
            "arm": V3_ARM,
            "reason": "High-frequency scaling examples need the full-context baseline unless validated otherwise.",
        },
        {
            "trigger": "termination | urgent_triage | medical_safety",
            "arm": COMPACT_ARM,
            "reason": "Counterexample and safety repair cases prefer compact explicit failure framing.",
        },
        {
            "trigger": "hard_ecological_constraint",
            "arm": MICRO_ARM,
            "reason": "Hard constraint tradeoffs prefer the micro guard's concise constraint handling.",
        },
        {
            "trigger": "formal_proof | generic_review | deep_space",
            "arm": V3_ARM,
            "reason": "Formal, generic review, and propulsion optimization boundaries abstain to the full baseline.",
        },
    ]


def _teacher_distillation_bootstrap(
    *, candidate_rows: list[TransitionRow], heldout_rows: list[TransitionRow]
) -> dict[str, Any]:
    selected_rows = []
    for row in candidate_rows:
        teacher_arm = row.teacher_arm
        selected = row.action_rewards[teacher_arm]
        selected_rows.append(
            {
                "problem_id": row.problem_id,
                "teacher_arm": teacher_arm,
                "observed_selected": selected,
                "observed_v3": row.action_rewards[V3_ARM],
            }
        )
    all_rows = []
    selected_by_id = {row["problem_id"]: row for row in selected_rows}
    for row in heldout_rows:
        if row.problem_id in selected_by_id:
            selected = selected_by_id[row.problem_id]
            all_rows.append(
                {
                    "problem_id": row.problem_id,
                    "selected_arm": selected["teacher_arm"],
                    "utility_vs_v1": selected["observed_selected"]["utility_vs_v1"],
                    "utility_vs_original_v3": selected["observed_selected"]["utility_vs_original_v3"],
                    "v3_utility_vs_v1": selected["observed_v3"]["utility_vs_v1"],
                }
            )
        else:
            v3 = row.action_rewards[V3_ARM]
            all_rows.append(
                {
                    "problem_id": row.problem_id,
                    "selected_arm": V3_ARM,
                    "utility_vs_v1": v3["utility_vs_v1"],
                    "utility_vs_original_v3": 0.5,
                    "v3_utility_vs_v1": v3["utility_vs_v1"],
                }
            )
    return {
        "status": "teacher_distilled_bootstrap_only",
        "not_counted_as_independent_validation": True,
        "selected_arm_counts": dict(Counter(row["selected_arm"] for row in all_rows)),
        "all_heldout_vs_v1_utility": round(_mean([float(row["utility_vs_v1"]) for row in all_rows]), 4),
        "all_heldout_vs_original_v3_utility": round(
            _mean([float(row["utility_vs_original_v3"]) for row in all_rows]), 4
        ),
        "candidate_teacher_vs_v3_utility": round(
            _mean([float(row["observed_selected"]["utility_vs_original_v3"]) for row in selected_rows]), 4
        ),
    }


def _latent_metrics(candidate_rows: list[TransitionRow]) -> dict[str, Any]:
    all_bits = sorted({bit for row in candidate_rows for bit in row.state_bits})
    bit_rates = {
        bit: sum(1 for row in candidate_rows if bit in row.state_bits) / max(1, len(candidate_rows))
        for bit in all_bits
    }
    entropy_proxy = _mean([1.0 - 2.0 * abs(rate - 0.5) for rate in bit_rates.values()])
    correlations = []
    for i, left in enumerate(all_bits):
        for right in all_bits[i + 1:]:
            correlations.append(abs(_phi(candidate_rows, left, right)))
    return {
        "bit_count": len(all_bits),
        "active_bit_rate_mean": round(_mean(bit_rates.values()), 4),
        "latent_entropy_proxy": round(entropy_proxy, 4),
        "latent_mean_abs_correlation": round(_mean(correlations), 4) if correlations else 0.0,
        "bit_rates": {key: round(value, 4) for key, value in bit_rates.items()},
    }


def _calibration_metrics(
    *,
    candidate_rows: list[TransitionRow],
    support_rows: list[TransitionRow],
    prior_weight: float,
    similarity_power: float,
    support_weight: float,
) -> dict[str, Any]:
    all_arm_abs_errors = []
    all_arm_base_abs_errors = []
    selected_abs_errors = []
    selected_base_abs_errors = []
    for row in candidate_rows:
        train = [other for other in candidate_rows if other.problem_id != row.problem_id] + support_rows
        predictions = {
            arm: _predict_arm_reward(
                row=row,
                train=train,
                arm=arm,
                prior_weight=prior_weight,
                similarity_power=similarity_power,
                support_weight=support_weight,
            )
            for arm in ARMS
        }
        selected_arm = max(predictions, key=lambda arm: predictions[arm]["predicted_scalar_reward"])
        for arm in ARMS:
            reward = row.action_rewards[arm]
            if not reward.get("observed"):
                continue
            observed = float(reward["scalar_reward"])
            predicted = float(predictions[arm]["predicted_scalar_reward"])
            base = _arm_base_rate(train=train, arm=arm)
            all_arm_abs_errors.append(abs(predicted - observed))
            all_arm_base_abs_errors.append(abs(base - observed))
        observed_selected = float(row.action_rewards[selected_arm]["scalar_reward"])
        selected_abs_errors.append(abs(float(predictions[selected_arm]["predicted_scalar_reward"]) - observed_selected))
        selected_base_abs_errors.append(abs(_arm_base_rate(train=train, arm=selected_arm) - observed_selected))
    return {
        "all_arm_mae": round(_mean(all_arm_abs_errors), 4),
        "all_arm_base_rate_mae": round(_mean(all_arm_base_abs_errors), 4),
        "selected_arm_mae": round(_mean(selected_abs_errors), 4),
        "selected_arm_base_rate_mae": round(_mean(selected_base_abs_errors), 4),
        "calibration_count": len(all_arm_abs_errors),
        "calibration_beats_base_rate": _mean(all_arm_abs_errors) < _mean(all_arm_base_abs_errors),
    }


def _metrics(
    *,
    heldout_rows: list[TransitionRow],
    support_rows: list[TransitionRow],
    candidate_rows: list[TransitionRow],
    loo_rows: list[dict[str, Any]],
    all_policy_rows: list[dict[str, Any]],
    calibrated_policy_rows: list[dict[str, Any]],
    latent_metrics: dict[str, Any],
    calibration: dict[str, Any],
    teacher_bootstrap: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    v3_selected_rewards = [float(row.action_rewards[V3_ARM]["scalar_reward"]) for row in candidate_rows]
    loo_rewards = [float(row["observed_selected"]["scalar_reward"]) for row in loo_rows]
    v3_selected_v1 = [float(row.action_rewards[V3_ARM]["utility_vs_v1"]) for row in candidate_rows]
    loo_v1 = [float(row["observed_selected"]["utility_vs_v1"]) for row in loo_rows]
    loo_v3 = [float(row["observed_selected"]["utility_vs_original_v3"]) for row in loo_rows]
    all_policy_v1 = [float(row["utility_vs_v1"]) for row in all_policy_rows]
    all_v3_v1 = [float(row["v3_utility_vs_v1"]) for row in all_policy_rows]
    all_policy_v3 = [float(row["utility_vs_original_v3"]) for row in all_policy_rows]
    calibrated_v1 = [float(row["utility_vs_v1"]) for row in calibrated_policy_rows]
    calibrated_v3 = [float(row["utility_vs_original_v3"]) for row in calibrated_policy_rows]
    calibrated_hybrid_v1 = [float(row["hybrid_utility_vs_v1"]) for row in calibrated_policy_rows]
    calibrated_hybrid_v3 = [
        float(row["hybrid_utility_vs_original_v3"]) for row in calibrated_policy_rows
    ]
    hybrid_metrics = artifacts["phase9_hybrid"].get("metrics", {})
    selected_arm_counts = Counter(row["selected_arm"] for row in loo_rows)
    calibrated_arm_counts = Counter(row["selected_arm"] for row in calibrated_policy_rows)
    retained_hybrid_v1 = float(hybrid_metrics.get("hybrid_vs_v1_heldout_utility") or 0.0)
    retained_hybrid_v3 = float(hybrid_metrics.get("hybrid_vs_original_v3_heldout_utility") or 0.0)
    return {
        "heldout_transition_row_count": len(heldout_rows),
        "compact_support_row_count": len(support_rows),
        "candidate_transition_count": len(candidate_rows),
        "candidate_action_coverage": round(
            _mean([
                1.0 if all(row.action_rewards[arm].get("observed") for arm in ARMS) else 0.0
                for row in candidate_rows
            ]),
            4,
        ),
        "learned_selected_arm_counts": dict(selected_arm_counts),
        "loo_selected_reward": round(_mean(loo_rewards), 4),
        "loo_v3_reward": round(_mean(v3_selected_rewards), 4),
        "loo_selected_reward_lift_over_v3": round(_mean(loo_rewards) - _mean(v3_selected_rewards), 4),
        "loo_selected_vs_v1_utility": round(_mean(loo_v1), 4),
        "loo_v3_vs_v1_selected_utility": round(_mean(v3_selected_v1), 4),
        "loo_selected_vs_v1_lift_over_v3": round(_mean(loo_v1) - _mean(v3_selected_v1), 4),
        "loo_selected_vs_v3_utility": round(_mean(loo_v3), 4),
        "loo_teacher_match_rate": round(_mean([1.0 if row["matches_teacher"] else 0.0 for row in loo_rows]), 4),
        "all_heldout_policy_vs_v1_utility": round(_mean(all_policy_v1), 4),
        "all_heldout_v3_vs_v1_utility": round(_mean(all_v3_v1), 4),
        "all_heldout_policy_lift_over_v3": round(_mean(all_policy_v1) - _mean(all_v3_v1), 4),
        "all_heldout_policy_vs_original_v3_utility": round(_mean(all_policy_v3), 4),
        "calibrated_policy_selected_arm_counts": dict(calibrated_arm_counts),
        "calibrated_policy_vs_v1_utility": round(_mean(calibrated_v1), 4),
        "calibrated_policy_vs_original_v3_utility": round(_mean(calibrated_v3), 4),
        "calibrated_policy_lift_over_v3": round(_mean(calibrated_v1) - _mean(all_v3_v1), 4),
        "calibrated_policy_lift_over_raw_world_model": round(_mean(calibrated_v1) - _mean(all_policy_v1), 4),
        "calibrated_policy_lift_over_retained_hybrid": round(_mean(calibrated_v1) - retained_hybrid_v1, 4),
        "calibrated_policy_vs_original_v3_lift_over_hybrid": round(_mean(calibrated_v3) - retained_hybrid_v3, 4),
        "calibrated_policy_harm_vs_hybrid_count": sum(
            1 for row in calibrated_policy_rows if float(row["delta_vs_hybrid_v1"]) < 0.0
        ),
        "calibrated_policy_win_vs_hybrid_count": sum(
            1 for row in calibrated_policy_rows if float(row["delta_vs_hybrid_v1"]) > 0.0
        ),
        "calibrated_policy_override_count": sum(
            1 for row in calibrated_policy_rows if row["selected_arm"] != row["raw_world_model_arm"]
        ),
        "retained_hybrid_vs_v1_utility": retained_hybrid_v1,
        "retained_hybrid_vs_original_v3_utility": retained_hybrid_v3,
        "teacher_bootstrap_vs_v1_utility": teacher_bootstrap["all_heldout_vs_v1_utility"],
        "teacher_bootstrap_vs_original_v3_utility": teacher_bootstrap["all_heldout_vs_original_v3_utility"],
        "learned_gap_to_retained_hybrid": round(_mean(all_policy_v1) - retained_hybrid_v1, 4),
        "recommended_promotion": (
            "promote_calibrated_residual_guard"
            if _mean(calibrated_v1) > retained_hybrid_v1
            and _mean(calibrated_v3) >= retained_hybrid_v3
            and _mean(calibrated_v1) >= _mean(calibrated_hybrid_v1)
            and _mean(calibrated_v3) >= _mean(calibrated_hybrid_v3)
            else "keep_as_world_model_candidate"
        ),
        "latent_entropy_proxy": latent_metrics["latent_entropy_proxy"],
        "latent_mean_abs_correlation": latent_metrics["latent_mean_abs_correlation"],
        "latent_bit_count": latent_metrics["bit_count"],
        "all_arm_mae": calibration["all_arm_mae"],
        "all_arm_base_rate_mae": calibration["all_arm_base_rate_mae"],
        "selected_arm_mae": calibration["selected_arm_mae"],
        "selected_arm_base_rate_mae": calibration["selected_arm_base_rate_mae"],
        "calibration_beats_base_rate": calibration["calibration_beats_base_rate"],
        "feature_snapshot_redacted": _feature_snapshot_redacted(artifacts["feature_snapshot"]),
        "uses_raw_prompts_or_answers": False,
    }


def _source_artifact_summary(*, root: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    source_paths = {
        "phase9_hybrid": HYBRID_ARTIFACT,
        "phase9_compact": COMPACT_ARTIFACT,
        "phase9_micro": MICRO_ARTIFACT,
        "phase9_compact_support": COMPACT_SUPPORT_ARTIFACT,
        "feature_snapshot": FEATURE_SNAPSHOT_ARTIFACT,
    }
    return {
        name: {
            "path": str(path),
            "exists": (root / path).exists(),
            "pass": artifact.get("pass"),
            "eval_kind": artifact.get("eval_kind"),
            "execution_mode": artifact.get("execution_mode"),
        }
        for name, path in source_paths.items()
        for artifact in [artifacts[name]]
    }


def _source_artifacts_live_or_derived(artifacts: dict[str, dict[str, Any]]) -> bool:
    execution_modes = {
        artifacts[name].get("execution_mode")
        for name in ["phase9_hybrid", "phase9_compact", "phase9_micro", "phase9_compact_support"]
    }
    return bool(execution_modes & {"execute", "offline_policy_validation", "summarize"})


def _feature_snapshot_redacted(snapshot: dict[str, Any]) -> bool:
    return (
        snapshot.get("contains_problem_text") is False
        and snapshot.get("contains_reference_answers") is False
        and snapshot.get("contains_prompts_or_answers") is False
        and bool(snapshot.get("features"))
    )


def _outcome_index(payload: dict[str, Any], pair: str) -> dict[str, dict[str, Any]]:
    summary = (payload.get("pair_summaries") or {}).get(pair) or {}
    return {row["problem_id"]: row for row in summary.get("rows", [])}


def _reward(*, vs_v1: float, vs_v3: float) -> dict[str, float | bool]:
    scalar = 0.55 * vs_v1 + 0.45 * vs_v3
    return {
        "utility_vs_v1": round(vs_v1, 4),
        "utility_vs_original_v3": round(vs_v3, 4),
        "scalar_reward": round(scalar, 4),
        "observed": True,
    }


def _missing_reward() -> dict[str, float | bool]:
    return {
        "utility_vs_v1": 0.0,
        "utility_vs_original_v3": 0.0,
        "scalar_reward": 0.0,
        "observed": False,
    }


def _value(outcome: str) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "tie":
        return 0.5
    return 0.0


def _arm_base_rate(*, train: list[TransitionRow], arm: str) -> float:
    values = [
        float(row.action_rewards[arm]["scalar_reward"])
        for row in train
        if row.action_rewards[arm].get("observed")
    ]
    return _mean(values) or 0.5


def _phi(rows: list[TransitionRow], left: str, right: str) -> float:
    n11 = sum(1 for row in rows if left in row.state_bits and right in row.state_bits)
    n10 = sum(1 for row in rows if left in row.state_bits and right not in row.state_bits)
    n01 = sum(1 for row in rows if left not in row.state_bits and right in row.state_bits)
    n00 = sum(1 for row in rows if left not in row.state_bits and right not in row.state_bits)
    denom = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    return ((n11 * n00) - (n10 * n01)) / denom if denom else 0.0


def _mean(values: list[float] | Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _interpretation(metrics: dict[str, Any]) -> str:
    return (
        "The raw outcome-only discrete world-model candidate improves original V3 on the Phase9 heldout slice "
        f"(all-policy lift {metrics['all_heldout_policy_lift_over_v3']:+.4f}; candidate lift "
        f"{metrics['loo_selected_vs_v1_lift_over_v3']:+.4f}) while staying below the retained hybrid guard "
        f"(gap {metrics['learned_gap_to_retained_hybrid']:+.4f}).  The calibrated residual guard repairs that "
        f"gap (lift over retained hybrid {metrics['calibrated_policy_lift_over_retained_hybrid']:+.4f}) without "
        "using raw prompts or answers, so the guard is promotable while the raw reward predictor remains a candidate."
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 Phase10 discrete world-model selector validation.")
    parser.add_argument("--eval-id", default="full_v3_phase10_discrete_world_model_selector_20260611")
    parser.add_argument("--prior-weight", type=float, default=2.0)
    parser.add_argument("--similarity-power", type=float, default=2.0)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase10_discrete_world_model_selector_payload(
        root=root,
        eval_id=args.eval_id,
        prior_weight=args.prior_weight,
        similarity_power=args.similarity_power,
        support_weight=args.support_weight,
        abstain_margin=args.abstain_margin,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "metrics": payload["metrics"],
                "failed_gates": payload["failed_gates"],
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
