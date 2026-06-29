"""Semantic-ish fidelity checks for answer-time operator use.

The older application fidelity verifier checks whether slot cues appear.  This
module adds a stricter metadata-only layer: did the answer instantiate slots
with problem-relevant substance, and did the operator change selection when a
baseline comparison is available?
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .application_fidelity import SLOT_CUES
from .operator_specs import OperatorSpec


GENERIC_SLOT_WORDS = {
    "apply",
    "assumption",
    "baseline",
    "check",
    "choose",
    "constraint",
    "control",
    "decision",
    "evidence",
    "factor",
    "metric",
    "operator",
    "relation",
    "source",
    "target",
    "variable",
}


@dataclass(frozen=True)
class SemanticSlotCheck:
    slot: str
    cue_present: bool
    substantive: bool
    problem_relevant: bool
    anchor_overlap_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticOperatorFidelity:
    source_id: str
    required_slot_count: int
    substantive_slot_count: int
    problem_relevant_slot_count: int
    slot_substance_rate: float
    problem_relevance_rate: float
    decision_change_observed: bool
    decision_changed: bool | None
    semantic_pass: bool
    slot_checks: list[SemanticSlotCheck]
    raw_content_persisted: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slot_checks"] = [check.to_dict() for check in self.slot_checks]
        return payload


def audit_operator_semantic_fidelity(
    *,
    problem_text: str,
    answer_text: str,
    spec: OperatorSpec | dict[str, Any],
    decision_changed: bool | None = None,
    min_slot_substance: float = 0.6,
    min_problem_relevance: float = 0.5,
) -> SemanticOperatorFidelity:
    operator = _coerce_spec(spec)
    problem_tokens = _content_tokens(problem_text)
    answer_tokens = _content_tokens(answer_text)
    checks = [
        _semantic_slot_check(slot=slot, problem_tokens=problem_tokens, answer_text=answer_text, answer_tokens=answer_tokens)
        for slot in operator.required_output_slots
    ]
    substantive_count = sum(1 for check in checks if check.substantive)
    relevant_count = sum(1 for check in checks if check.problem_relevant)
    slot_rate = substantive_count / len(checks) if checks else 1.0
    relevance_rate = relevant_count / len(checks) if checks else 1.0
    change_ok = True if decision_changed is None else bool(decision_changed)
    semantic_pass = bool(
        slot_rate >= min_slot_substance
        and relevance_rate >= min_problem_relevance
        and change_ok
    )
    return SemanticOperatorFidelity(
        source_id=operator.source_id,
        required_slot_count=len(checks),
        substantive_slot_count=substantive_count,
        problem_relevant_slot_count=relevant_count,
        slot_substance_rate=round(slot_rate, 4),
        problem_relevance_rate=round(relevance_rate, 4),
        decision_change_observed=decision_changed is not None,
        decision_changed=decision_changed,
        semantic_pass=semantic_pass,
        slot_checks=checks,
    )


def audit_answer_semantic_fidelity(
    *,
    problem_text: str,
    answer_text: str,
    specs: Iterable[OperatorSpec | dict[str, Any]],
    decision_changed: bool | None = None,
) -> dict[str, Any]:
    audits = [
        audit_operator_semantic_fidelity(
            problem_text=problem_text,
            answer_text=answer_text,
            spec=spec,
            decision_changed=decision_changed,
        )
        for spec in specs
    ]
    if not audits:
        return {
            "operator_count": 0,
            "slot_substance_rate": 1.0,
            "problem_relevance_rate": 1.0,
            "decision_change_observed": decision_changed is not None,
            "decision_changed": decision_changed,
            "semantic_pass": True,
            "operators": [],
            "raw_content_persisted": False,
        }
    slot_rate = sum(audit.slot_substance_rate for audit in audits) / len(audits)
    relevance_rate = sum(audit.problem_relevance_rate for audit in audits) / len(audits)
    return {
        "operator_count": len(audits),
        "slot_substance_rate": round(slot_rate, 4),
        "problem_relevance_rate": round(relevance_rate, 4),
        "decision_change_observed": decision_changed is not None,
        "decision_changed": decision_changed,
        "semantic_pass": all(audit.semantic_pass for audit in audits),
        "operators": [audit.to_dict() for audit in audits],
        "raw_content_persisted": False,
    }


def _semantic_slot_check(
    *,
    slot: str,
    problem_tokens: set[str],
    answer_text: str,
    answer_tokens: set[str],
) -> SemanticSlotCheck:
    cues = _slot_cues(slot)
    answer_lower = str(answer_text or "").lower()
    cue_present = any(cue.lower() in answer_lower for cue in cues)
    anchor_overlap = len(problem_tokens & answer_tokens)
    slot_tokens = _content_tokens(slot.replace("_", " "))
    non_generic_answer_tokens = {
        token for token in answer_tokens
        if token not in GENERIC_SLOT_WORDS and token not in slot_tokens
    }
    has_concrete_anchor = bool(anchor_overlap >= 2 or _has_number_or_option(answer_text))
    substantive = bool(cue_present and non_generic_answer_tokens and has_concrete_anchor)
    problem_relevant = bool(substantive and (anchor_overlap >= 2 or problem_tokens & slot_tokens))
    return SemanticSlotCheck(
        slot=slot,
        cue_present=cue_present,
        substantive=substantive,
        problem_relevant=problem_relevant,
        anchor_overlap_count=anchor_overlap,
    )


def _slot_cues(slot: str) -> list[str]:
    cues = list(SLOT_CUES.get(slot, []))
    cues.extend(part for part in re.split(r"[_\W]+", str(slot or "").lower()) if part)
    seen: set[str] = set()
    out: list[str] = []
    for cue in cues:
        value = str(cue).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", str(text or "").lower()))
    return {token for token in tokens if token not in GENERIC_SLOT_WORDS}


def _has_number_or_option(text: str) -> bool:
    return bool(re.search(r"\b(?:[A-Z]|[0-9]+(?:\.[0-9]+)?)\b", str(text or "")))


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
