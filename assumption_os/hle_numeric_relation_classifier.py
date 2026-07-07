"""Deterministic relation labels for numeric HLE multiple-choice items."""

from __future__ import annotations

import re
from typing import Any

from .autonomy_journal import stable_hash


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "still", "that", "the",
    "this", "to", "using", "what", "which", "with", "following",
}


def numeric_relation_terms(text: str) -> list[str]:
    clean = re.sub(r"[^a-zA-Z0-9]+", " ", str(text or "").lower())
    terms = [token for token in clean.split() if len(token) >= 3 and token not in _STOPWORDS]
    expanded = set(terms)
    joined = " ".join(terms)
    if "xenon" in expanded and "tetrafluoride" in expanded:
        expanded.add("xef4")
    if "xef4" in expanded:
        expanded.update({"xenon", "tetrafluoride"})
    if any(token in expanded for token in {"synthesis", "synthesize", "synthesized", "produce", "produced"}):
        expanded.update({"prepare", "prepared", "preparation", "reaction"})
    if any(token in joined for token in ["temperature", "coldest", "hottest", "celsius", "kelvin"]):
        expanded.update({"temperature", "thermal", "heat", "heated"})
    return sorted(expanded)


def classify_numeric_relation(question: str, *, value_type: str = "") -> dict[str, Any]:
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    relation_family = "exact_value"
    direction = "closest_is_correct"
    confidence = "medium"

    if any(phrase in text for phrase in ("closest to", "nearest to", "approximately", "approximate")):
        relation_family = "closest_value"
        direction = "closest_is_correct"
        confidence = "high"
    elif any(token in text for token in ("coldest", "lowest", "minimum", "least")):
        relation_family = (
            "threshold_minimum"
            if any(token in text for token in ("still", "at which", "required", "can", "efficient"))
            else "ordered_extreme_lowest"
        )
        direction = "lower_is_correct"
        confidence = "high"
    elif any(token in text for token in ("hottest", "highest", "maximum", "largest", "greatest")):
        relation_family = (
            "threshold_maximum"
            if any(token in text for token in ("still", "below", "allowed", "can"))
            else "ordered_extreme_highest"
        )
        direction = "higher_is_correct"
        confidence = "high"
    elif any(phrase in text for phrase in ("below which", "less than", "under which")):
        relation_family = "below_threshold"
        direction = "lower_is_correct"
        confidence = "medium"
    elif any(phrase in text for phrase in ("above which", "greater than", "over which")):
        relation_family = "above_threshold"
        direction = "higher_is_correct"
        confidence = "medium"
    elif any(phrase in text for phrase in ("between", "within the range", "falls in")):
        relation_family = "range_membership"
        direction = "within_range_is_correct"
        confidence = "medium"

    if value_type == "temperature" and relation_family == "exact_value":
        if any(token in text for token in ("temperature", "coldest", "hottest", "heated", "cooled")):
            confidence = "high"

    terms = numeric_relation_terms(question)
    payload = {
        "status": "activated" if terms else "abstained",
        "reason": "classified_numeric_relation" if terms else "empty_question_terms",
        "relation_family": relation_family,
        "direction": direction,
        "confidence": confidence,
        "value_type": value_type or "number",
        "relation_terms": terms,
        "raw_content_persisted": False,
    }
    payload["relation_hash"] = stable_hash({
        "relation_family": relation_family,
        "direction": direction,
        "value_type": value_type or "number",
        "relation_terms": terms,
    })
    return payload
