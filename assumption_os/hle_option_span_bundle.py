"""Deterministic source-span bundle features for option-centered routing."""

from __future__ import annotations

import re
from typing import Any

from .autonomy_journal import stable_hash


_STOPWORDS = {
    "about",
    "after",
    "also",
    "answer",
    "based",
    "because",
    "before",
    "being",
    "between",
    "choose",
    "correct",
    "could",
    "difference",
    "during",
    "following",
    "from",
    "have",
    "into",
    "more",
    "most",
    "option",
    "question",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "under",
    "what",
    "when",
    "which",
    "with",
    "would",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+-]{2,}", str(text or "").lower())
        if token not in _STOPWORDS
    ]


def _terms(text: str, *, limit: int = 24) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in _tokens(text):
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _record_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""
    parts: list[str] = []
    for key in ("span_text", "snippet", "abstract", "title", "text", "content"):
        value = str(record.get(key) or "").strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def _infer_relation_terms(question: str) -> list[str]:
    normalized = str(question or "").lower()
    relation_cues = [
        "cause",
        "causes",
        "caused",
        "leads",
        "lead",
        "increase",
        "increases",
        "decrease",
        "decreases",
        "stronger",
        "weaker",
        "bind",
        "binds",
        "binding",
        "fluorescent",
        "fluorescence",
        "signal",
        "probe",
        "click",
        "azide",
        "alkyne",
        "mass",
        "peak",
        "formula",
    ]
    return [cue for cue in relation_cues if cue in normalized]


def build_option_span_bundles(
    *,
    question: str,
    options: dict[str, str],
    source_records_by_option: dict[str, list[Any]] | None = None,
    relation_terms: list[str] | None = None,
    required_terms: list[str] | None = None,
    top_k: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Build fixed top-k span bundles per option from already available sources."""

    relation_terms = relation_terms or _infer_relation_terms(question)
    required_terms = required_terms or []
    source_records_by_option = source_records_by_option or {}
    output: dict[str, list[dict[str, Any]]] = {}
    for label, option_text in sorted(options.items()):
        option_terms = _terms(option_text)
        records = list(source_records_by_option.get(label, []) or [])
        bundles: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            text = _record_text(record)
            if not text.strip():
                continue
            text_terms = set(_terms(text, limit=128))
            option_overlap = sorted(term for term in option_terms if term in text_terms)
            relation_overlap = sorted(term for term in relation_terms if term.lower() in text.lower())
            required_present = sorted(term for term in required_terms if term.lower() in text.lower())
            required_missing = sorted(term for term in required_terms if term.lower() not in text.lower())
            shared_doc_option_count = 0
            if isinstance(record, dict):
                try:
                    shared_doc_option_count = int(record.get("shared_doc_option_count") or 0)
                except (TypeError, ValueError):
                    shared_doc_option_count = 0
            generic_penalty = 0.2 if not option_overlap else 0.0
            shared_doc_penalty = min(0.4, max(0, shared_doc_option_count - 1) * 0.1)
            directness_score = (
                (0.45 if option_overlap else 0.0)
                + (0.30 if relation_overlap else 0.0)
                + (0.20 if required_terms and not required_missing else 0.0)
                - generic_penalty
                - shared_doc_penalty
            )
            if option_overlap and relation_overlap and (not required_terms or not required_missing):
                bundle_type = "direct_relation"
            elif option_overlap and relation_overlap:
                bundle_type = "indirect"
            elif option_overlap:
                bundle_type = "definition"
            else:
                bundle_type = "generic"
            bundle = {
                "option_label": label,
                "option_hash": stable_hash({"option_label": label}),
                "source_id_hash": stable_hash({
                    "source_id": (record.get("source_id") if isinstance(record, dict) else index)
                }),
                "span_hash": stable_hash({"span_text": text}),
                "option_overlap_terms": option_overlap,
                "anchor_overlap_terms": [],
                "relation_overlap_terms": relation_overlap,
                "required_terms_present": required_present,
                "required_terms_missing": required_missing,
                "shared_doc_option_count": shared_doc_option_count,
                "shared_doc_penalty": round(shared_doc_penalty, 3),
                "generic_penalty": round(generic_penalty, 3),
                "bundle_type": bundle_type,
                "directness_score": round(max(0.0, min(1.0, directness_score)), 3),
                "raw_content_persisted": False,
            }
            bundle["bundle_hash"] = stable_hash({
                "option_hash": bundle["option_hash"],
                "source_id_hash": bundle["source_id_hash"],
                "span_hash": bundle["span_hash"],
                "bundle_type": bundle["bundle_type"],
                "directness_score": bundle["directness_score"],
            })
            bundles.append(bundle)
        output[label] = sorted(
            bundles,
            key=lambda bundle: (
                -float(bundle.get("directness_score") or 0.0),
                str(bundle.get("bundle_hash") or ""),
            ),
        )[: max(1, int(top_k or 1))]
    return output
