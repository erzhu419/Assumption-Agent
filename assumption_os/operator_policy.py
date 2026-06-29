"""Operator selection policy for answer-time assumption operators.

OperatorSpecs describe how to execute a retrieved assumption.  This module adds
the missing policy layer: whether the operator should be used, how strong it
should be, and when it should abstain.  It is deterministic and metadata-only so
HLE runs can log policy decisions without persisting question text.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .operator_specs import OperatorSpec


POLICY_VERSION = "operator_policy_generalization_v1"

OPERATOR_STRENGTHS = {"off", "soft", "required", "strict", "repair"}

FAMILY_CUES: dict[str, dict[str, set[str]]] = {
    "O1_causal_control_variable": {
        "spec": {"control", "controlled", "causal", "cause", "variable", "baseline", "confound"},
        "problem": {"control", "controlled", "cause", "causal", "effect", "variable", "baseline", "treatment"},
    },
    "O2_dependency_aware_intervention": {
        "spec": {"dependency", "coupling", "coupled", "confound", "intervention", "contrast"},
        "problem": {"dependency", "coupled", "coupling", "interaction", "intervention", "resistance", "mutation"},
    },
    "O3_incremental_replacement": {
        "spec": {"incremental", "migration", "legacy", "replace", "replacement", "rollback", "strangler"},
        "problem": {"incremental", "migration", "legacy", "replace", "replacement", "rollout", "upgrade"},
    },
    "O4_adapter_boundary_discovery": {
        "spec": {"adapter", "boundary", "interface", "wrapper", "contract"},
        "problem": {"adapter", "boundary", "interface", "contract", "compatibility", "bridge"},
    },
    "O5_evidence_grounding": {
        "spec": {"evidence", "source", "retrieval", "relation", "answer-bearing", "scope"},
        "problem": {"evidence", "supported", "source", "relation", "claim", "statement", "true", "infer"},
    },
    "O6_analogy_role_mapping": {
        "spec": {"analogy", "mapping", "morphism", "role", "invariant", "source", "target"},
        "problem": {"analogous", "analogy", "mapping", "role", "invariant", "homology", "correspond"},
    },
    "O7_limiting_case_reduction": {
        "spec": {"limiting", "limit", "degenerate", "boundary", "reduction"},
        "problem": {"limit", "limiting", "degenerate", "boundary", "asymptotic"},
    },
    "O8_negative_control_abstention": {
        "spec": {"negative", "control", "abstain", "fallback", "not applicable"},
        "problem": {"except", "not", "cannot", "fail", "invalid", "counterexample"},
    },
    "O9_multi_objective_tradeoff": {
        "spec": {"tradeoff", "objective", "cost", "benefit"},
        "problem": {"tradeoff", "objective", "constraint", "cost", "benefit", "optimize"},
    },
    "O10_stakeholder_constraint_mapping": {
        "spec": {"stakeholder", "owner", "customer", "requirement"},
        "problem": {"stakeholder", "owner", "user", "customer", "requirement", "policy"},
    },
    "O11_failure_mode_triage": {
        "spec": {"failure", "risk", "triage", "diagnose", "fault"},
        "problem": {"failure", "risk", "bug", "error", "fault", "diagnose", "root"},
    },
    "O12_verification_plan_construction": {
        "spec": {"verify", "verifier", "test", "check", "validation", "metric"},
        "problem": {"verify", "test", "check", "validation", "metric", "measure", "evaluate"},
    },
}

CAUTION_DOMAINS = {"business", "software_engineering"}
TRUSTED_DOMAINS = {"daily_life"}


@dataclass(frozen=True)
class OperatorPolicyScore:
    source_id: str
    source_type: str
    p_trigger: float
    p_harm: float
    family_ids: list[str]
    selected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorPolicyDecision:
    policy_version: str
    enabled: bool
    domain: str
    operator_strength: str
    selected_operator_ids: list[str]
    selected_operator_family_ids: list[str]
    p_trigger: float
    p_harm: float
    abstain_reason: str | None
    spec_count: int
    score_rows: list[OperatorPolicyScore]
    gate_applied: bool = False
    raw_content_persisted: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score_rows"] = [row.to_dict() for row in self.score_rows]
        return payload


def decide_operator_policy(
    *,
    problem_text: str | dict[str, Any] = "",
    specs: Iterable[OperatorSpec | dict[str, Any]],
    domain: str = "",
    context_allowed: bool = True,
    generic_graph_context_only: bool = False,
    max_selected: int = 2,
) -> OperatorPolicyDecision:
    """Return a deterministic operator policy decision.

    The decision stores only ids, counts, probabilities, and taxonomy labels.
    It intentionally does not store the problem or answer text.
    """

    operator_specs = [_coerce_spec(spec) for spec in specs]
    dom = str(domain or "").strip()
    if not operator_specs:
        return OperatorPolicyDecision(
            policy_version=POLICY_VERSION,
            enabled=False,
            domain=dom,
            operator_strength="off",
            selected_operator_ids=[],
            selected_operator_family_ids=[],
            p_trigger=0.0,
            p_harm=0.0,
            abstain_reason="no_operator_specs",
            spec_count=0,
            score_rows=[],
        )

    problem_tokens = _tokens(_problem_to_text(problem_text))
    rows = [
        _score_spec(
            spec,
            problem_tokens=problem_tokens,
            domain=dom,
            context_allowed=context_allowed,
            generic_graph_context_only=generic_graph_context_only,
        )
        for spec in operator_specs
    ]
    rows = sorted(rows, key=lambda row: (row.selected, row.p_trigger - row.p_harm, row.p_trigger), reverse=True)
    selected_rows = [row for row in rows if row.selected][: max(1, max_selected)]
    selected_ids = [row.source_id for row in selected_rows]
    selected_families = sorted({family for row in selected_rows for family in row.family_ids})
    p_trigger = max((row.p_trigger for row in rows), default=0.0)
    p_harm = max((row.p_harm for row in rows), default=0.0)
    strength = _strength_from_scores(p_trigger, p_harm, selected_rows, domain=dom)
    abstain_reason = None
    if not selected_rows:
        abstain_reason = "policy_trigger_below_floor"
        strength = "off"
    elif p_harm >= p_trigger and dom in CAUTION_DOMAINS:
        abstain_reason = "policy_harm_exceeds_trigger_in_caution_domain"
        strength = "off"
        selected_ids = []
        selected_families = []
    return OperatorPolicyDecision(
        policy_version=POLICY_VERSION,
        enabled=bool(selected_ids),
        domain=dom,
        operator_strength=strength,
        selected_operator_ids=selected_ids,
        selected_operator_family_ids=selected_families,
        p_trigger=round(p_trigger, 4),
        p_harm=round(p_harm, 4),
        abstain_reason=abstain_reason,
        spec_count=len(operator_specs),
        score_rows=rows,
    )


def classify_operator_families(spec: OperatorSpec | dict[str, Any]) -> list[str]:
    operator = _coerce_spec(spec)
    spec_tokens = _tokens(_spec_text(operator))
    families = [
        family_id
        for family_id, cues in FAMILY_CUES.items()
        if spec_tokens & cues["spec"]
    ]
    return families or ["generic_operator"]


def _score_spec(
    spec: OperatorSpec,
    *,
    problem_tokens: set[str],
    domain: str,
    context_allowed: bool,
    generic_graph_context_only: bool,
) -> OperatorPolicyScore:
    spec_text = _spec_text(spec)
    spec_tokens = _tokens(spec_text)
    families = classify_operator_families(spec)
    problem_hit_count = 0
    for family in families:
        cues = FAMILY_CUES.get(family)
        if cues:
            problem_hit_count += len(problem_tokens & cues["problem"])
    generic = families == ["generic_operator"]
    slot_count = len(spec.required_output_slots)
    check_count = len(spec.verifier_checks)
    confidence = max(0.0, min(1.0, float(spec.confidence or 0.0)))
    builtin_bonus = 0.08 if str(spec.source_id).startswith("framework_") else 0.0
    domain_bonus = 0.08 if domain in TRUSTED_DOMAINS else 0.0
    context_penalty = 0.12 if generic_graph_context_only or not context_allowed else 0.0
    p_trigger = (
        0.18
        + 0.42 * confidence
        + min(0.24, 0.06 * problem_hit_count)
        + min(0.10, 0.02 * slot_count)
        + builtin_bonus
        + domain_bonus
        - context_penalty
    )
    harm = (
        0.08
        + (0.18 if generic else 0.0)
        + (0.10 if slot_count >= 6 else 0.0)
        + (0.04 if check_count >= 5 else 0.0)
        + (0.08 if domain in CAUTION_DOMAINS and generic else 0.0)
        + (0.10 if generic_graph_context_only else 0.0)
    )
    p_trigger = _bounded(p_trigger)
    p_harm = _bounded(harm)
    margin = p_trigger - p_harm
    selected = bool(p_trigger >= 0.38 and margin >= 0.05)
    if selected:
        reason = "trigger_beats_harm_margin"
    elif generic:
        reason = "generic_operator_without_specific_family"
    elif p_trigger < 0.38:
        reason = "trigger_below_floor"
    else:
        reason = "harm_margin_too_small"
    return OperatorPolicyScore(
        source_id=spec.source_id,
        source_type=spec.source_type,
        p_trigger=round(p_trigger, 4),
        p_harm=round(p_harm, 4),
        family_ids=families,
        selected=selected,
        reason=reason,
    )


def _strength_from_scores(
    p_trigger: float,
    p_harm: float,
    selected_rows: list[OperatorPolicyScore],
    *,
    domain: str,
) -> str:
    if not selected_rows:
        return "off"
    margin = p_trigger - p_harm
    if p_trigger >= 0.82 and margin >= 0.45 and domain in TRUSTED_DOMAINS:
        return "repair"
    if p_trigger >= 0.74 and margin >= 0.28:
        return "strict"
    if p_trigger >= 0.58 and margin >= 0.16:
        return "required"
    return "soft"


def _coerce_spec(spec: OperatorSpec | dict[str, Any]) -> OperatorSpec:
    if isinstance(spec, OperatorSpec):
        return spec
    payload = dict(spec)
    payload.setdefault("source_id", "")
    payload.setdefault("source_type", "")
    payload.setdefault("source_claim", "")
    payload.setdefault("trigger_conditions", [])
    payload.setdefault("execution_steps", [])
    payload.setdefault("required_output_slots", [])
    payload.setdefault("negative_controls", [])
    payload.setdefault("verifier_checks", [])
    payload.setdefault("fallback_policy", "")
    payload.setdefault("confidence", 0.0)
    return OperatorSpec(**payload)


def _problem_to_text(problem_text: str | dict[str, Any]) -> str:
    if isinstance(problem_text, dict):
        return " ".join(
            str(problem_text.get(key) or "")
            for key in ("_question", "question", "subject", "answer_type")
        )
    return str(problem_text or "")


def _spec_text(spec: OperatorSpec) -> str:
    return " ".join([
        spec.source_id,
        spec.source_type,
        spec.source_claim,
        " ".join(spec.trigger_conditions),
        " ".join(spec.execution_steps),
        " ".join(spec.required_output_slots),
        " ".join(spec.negative_controls),
        " ".join(spec.verifier_checks),
        spec.fallback_policy,
    ]).lower()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_-]*", str(text or "").lower()))


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
