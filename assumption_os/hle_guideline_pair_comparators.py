"""Programmatic source comparators for guideline/order and pair-binding probes.

These helpers are diagnostic-first.  They intentionally return only hashes,
counts, and compact scores so source-prefetch artifacts do not persist raw HLE
question or option text.
"""

from __future__ import annotations

import re
from typing import Any

from .autonomy_journal import stable_hash


def _text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("title", "snippet", "text", "abstract")
        if str(row.get(key) or "").strip()
    )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.sub(r"[^A-Za-z0-9+]+", " ", str(text or "").lower()).split()
        if len(token) >= 2
    }


def _patient_descriptions(stem: str) -> dict[str, str]:
    text = re.sub(r"\s+", " ", str(stem or " ").replace("\n", " ")).strip()
    out: dict[str, str] = {}
    pattern = re.compile(
        r"\bPatient\s+(\d{1,2})\s*:\s*(.*?)(?=\bPatient\s+\d{1,2}\s*:|\bAnswer\s+Choices\b|\bPrioriti[sz]e\b|\bWhich\b|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        label = match.group(1).strip()
        description = re.sub(r"\s+", " ", match.group(2)).strip(" .;:")
        if label and description:
            out[label] = description
    return out


def _patient_order(option_text: str) -> list[str]:
    return list(dict.fromkeys(
        match.group(1)
        for match in re.finditer(
            r"\bPatient\s+(\d{1,2})\b",
            str(option_text or ""),
            flags=re.IGNORECASE,
        )
    ))


def _source_hashes(rows: list[dict[str, Any]]) -> list[str]:
    return [
        stable_hash({
            "title": str(row.get("title") or ""),
            "snippet": str(row.get("snippet") or ""),
            "source": str(row.get("source") or ""),
        })
        for row in rows[:5]
        if isinstance(row, dict)
    ]


def _medical_guideline_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_terms = {
        "classification",
        "guideline",
        "morphology",
        "neurologic",
        "operative",
        "patient",
        "points",
        "score",
        "surgical",
        "thoracolumbar",
        "tlics",
        "trauma",
        "treatment",
    }
    row_count = 0
    best_overlap = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_terms = _tokens(_text(row))
        overlap = len(doc_terms & evidence_terms)
        if overlap >= 2 and (doc_terms & {"thoracolumbar", "tlics", "spine", "trauma"}):
            row_count += 1
        best_overlap = max(best_overlap, overlap)
    return {
        "evidence_row_count": row_count,
        "best_evidence_overlap": best_overlap,
    }


def _patient_guideline_score(description: str) -> dict[str, Any]:
    text = str(description or "").lower()
    morphology_score = 0
    morphology_reason = "unknown"
    if re.search(r"\bdistraction\b", text):
        morphology_score = 4
        morphology_reason = "distraction"
    elif re.search(r"\b(translat|rotat|spondylolisthesis|dislocation)\w*", text):
        morphology_score = 3
        morphology_reason = "translation_or_spondylolisthesis"
    elif re.search(r"\bburst\b", text):
        morphology_score = 2
        morphology_reason = "burst"
    elif re.search(r"\bsplit\b", text):
        morphology_score = 2
        morphology_reason = "split_burst_like"
    elif re.search(r"\bcompression\b", text):
        morphology_score = 1
        morphology_reason = "compression"

    neurologic_score = 0
    neurologic_reason = "intact_or_not_reported"
    if re.search(r"\b(no|without)\s+neurologic", text):
        neurologic_score = 0
        neurologic_reason = "explicit_no_neurologic_deficit"
    elif re.search(r"\b(cauda|conus|pelvic|bowel|bladder|incomplete|disordered)\b", text):
        neurologic_score = 3
        neurologic_reason = "pelvic_or_incomplete_neurologic_signal"
    elif re.search(r"\bnerve\s+root\b", text):
        neurologic_score = 2
        neurologic_reason = "nerve_root_signal"
    elif re.search(r"\bneurologic", text):
        neurologic_score = 2
        neurologic_reason = "neurologic_signal"

    total = morphology_score + neurologic_score
    return {
        "score": total,
        "morphology_score": morphology_score,
        "morphology_reason": morphology_reason,
        "neurologic_score": neurologic_score,
        "neurologic_reason": neurologic_reason,
    }


def medical_guideline_permutation_ordering_detail(
    *,
    stem: str,
    option_text: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = "medical_guideline_permutation_ordering_v1"
    descriptions = _patient_descriptions(stem)
    candidate_order = _patient_order(option_text)
    base = {
        "policy": policy,
        "status": "not_applicable",
        "reason": "not_patient_permutation_guideline_question",
        "raw_content_persisted": False,
    }
    if len(descriptions) < 2 or len(candidate_order) < 2:
        return base
    question_text = str(stem or "").lower()
    if not (
        "patient" in question_text
        and (
            "surgical" in question_text
            or "indication" in question_text
            or "prioritize" in question_text
        )
    ):
        return base
    evidence = _medical_guideline_evidence(rows)
    if int(evidence.get("evidence_row_count") or 0) <= 0:
        return {
            **base,
            "status": "blocked",
            "reason": "missing_guideline_source_evidence",
            "candidate_order_hash": stable_hash({"patient_order": candidate_order}),
            "patient_count": len(descriptions),
            "source_hashes": _source_hashes(rows),
            **evidence,
        }

    score_rows = []
    for label, description in descriptions.items():
        score = _patient_guideline_score(description)
        score_rows.append({
            "patient_hash": stable_hash({"patient_label": label}),
            "label": label,
            "score": score["score"],
            "morphology_score": score["morphology_score"],
            "morphology_reason": score["morphology_reason"],
            "neurologic_score": score["neurologic_score"],
            "neurologic_reason": score["neurologic_reason"],
            "description_hash": stable_hash({"patient_description": description}),
        })
    score_by_label = {str(row["label"]): int(row["score"]) for row in score_rows}
    expected_order = [
        row["label"]
        for row in sorted(
            score_rows,
            key=lambda row: (-int(row["score"]), str(row["label"])),
        )
    ]
    score_values = sorted(score_by_label.values(), reverse=True)
    ambiguous = any(
        left == right
        for left, right in zip(score_values, score_values[1:])
    )
    candidate_known = all(label in score_by_label for label in candidate_order)
    exact_match = bool(candidate_known and candidate_order == expected_order[:len(candidate_order)])
    rank_penalty = 0
    if candidate_known:
        expected_rank = {label: index for index, label in enumerate(expected_order)}
        rank_penalty = sum(
            abs(index - expected_rank.get(label, index))
            for index, label in enumerate(candidate_order)
        )
    candidate_score = (
        sum(
            max(0, len(candidate_order) - index) * score_by_label.get(label, 0)
            for index, label in enumerate(candidate_order)
        )
        if candidate_known
        else 0
    )
    status = "activated" if exact_match and not ambiguous else "evaluated"
    reason = (
        "candidate_order_matches_guideline_severity"
        if exact_match and not ambiguous
        else (
            "ambiguous_patient_guideline_scores"
            if ambiguous
            else "candidate_order_not_guideline_severity_order"
        )
    )
    return {
        **base,
        "status": status,
        "reason": reason,
        "patient_count": len(descriptions),
        "candidate_known": candidate_known,
        "candidate_order_hash": stable_hash({"patient_order": candidate_order}),
        "expected_order_hash": stable_hash({"patient_order": expected_order}),
        "candidate_exact_expected_order": exact_match,
        "candidate_rank_penalty": rank_penalty,
        "candidate_guideline_order_score": candidate_score,
        "ambiguous_patient_scores": ambiguous,
        "score_rows": score_rows,
        "source_hashes": _source_hashes(rows),
        **evidence,
    }


_FE_OXIDATION_ALIASES = {
    "I": {"fe1", "fe+", "fe 1+", "iron i"},
    "II": {"fe2", "fe2+", "fe 2+", "ferrous", "iron ii", "iron(ii)"},
    "III": {"fe3", "fe3+", "fe 3+", "ferric", "iron iii", "iron(iii)"},
    "IV": {"fe4", "fe4+", "fe 4+", "iron iv", "iron(iv)"},
    "V": {"fe5", "fe5+", "fe 5+", "iron v", "iron(v)"},
    "VI": {"fe6", "fe6+", "fe 6+", "iron vi", "iron(vi)"},
}


def _fe_option_features(option_text: str) -> dict[str, Any]:
    text = str(option_text or "")
    lower = text.lower()
    oxidation = ""
    match = re.search(r"\bFe\s*\(\s*([IVX]+)\s*\)", text, flags=re.IGNORECASE)
    if match:
        oxidation = match.group(1).upper()
    spin = ""
    spin_match = re.search(r"\bS\s*=\s*([0-9]+(?:/[0-9]+)?(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if spin_match:
        spin = spin_match.group(1).strip()
    geometries = []
    for phrase in (
        "square pyramidal",
        "trigonal bipyramidal",
        "tetrahedral",
        "octahedral",
        "linear",
        "planar",
    ):
        if phrase in lower:
            geometries.append(phrase)
    return {
        "oxidation": oxidation,
        "spin": spin,
        "geometry_terms": geometries,
        "option_feature_hash": stable_hash({
            "oxidation": oxidation,
            "spin": spin,
            "geometry_terms": geometries,
        }),
    }


def _contains_alias(text_lower: str, aliases: set[str]) -> bool:
    compact = re.sub(r"\s+", " ", text_lower)
    no_space = compact.replace(" ", "")
    for alias in aliases:
        alias_lower = alias.lower()
        if alias_lower in compact or alias_lower.replace(" ", "") in no_space:
            return True
    return False


def _contains_spin(text_lower: str, spin: str) -> bool:
    if not spin:
        return False
    normalized = spin.replace("/", r"\s*/\s*")
    return bool(
        re.search(rf"\bS\s*=\s*{normalized}\b", text_lower, flags=re.IGNORECASE)
        or re.search(rf"\bspin\s+{normalized}\b", text_lower, flags=re.IGNORECASE)
    )


def fe_hyperfine_pair_binding_detail(
    *,
    stem: str,
    option_text: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = "fe_hyperfine_pair_binding_v1"
    base = {
        "policy": policy,
        "status": "not_applicable",
        "reason": "not_fe_hyperfine_question",
        "raw_content_persisted": False,
    }
    question = str(stem or "").lower()
    if not ("hyperfine" in question and ("mossbauer" in question or "57fe" in question)):
        return base
    features = _fe_option_features(option_text)
    oxidation = str(features.get("oxidation") or "")
    if not oxidation:
        return base
    aliases = _FE_OXIDATION_ALIASES.get(oxidation, set())
    spin = str(features.get("spin") or "")
    geometry_terms = [str(term) for term in features.get("geometry_terms", []) or []]

    row_details: list[dict[str, Any]] = []
    direct_count = 0
    partial_count = 0
    missing_geometry_count = 0
    missing_relation_count = 0
    best_score = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        text_lower = _text(row).lower()
        if not text_lower:
            continue
        oxidation_bound = _contains_alias(text_lower, aliases)
        spin_bound = _contains_spin(text_lower, spin)
        geometry_bound = bool(
            not geometry_terms
            or any(term.lower() in text_lower for term in geometry_terms)
        )
        hyperfine_bound = "hyperfine" in text_lower
        relation_bound = bool(
            hyperfine_bound
            and (
                "field" in text_lower
                or "magnetic interaction" in text_lower
                or "magnetic hyperfine" in text_lower
            )
        )
        superlative_bound = bool(
            "largest" in text_lower
            or "highest" in text_lower
            or "greater" in text_lower
            or "large" in text_lower
        )
        pair_bound = bool(oxidation_bound and spin_bound)
        direct = bool(pair_bound and geometry_bound and relation_bound and superlative_bound)
        partial = bool(pair_bound and relation_bound)
        if direct:
            direct_count += 1
        if partial:
            partial_count += 1
        if partial and not geometry_bound:
            missing_geometry_count += 1
        if pair_bound and not relation_bound:
            missing_relation_count += 1
        score = (
            (2.0 if oxidation_bound else 0.0)
            + (2.0 if spin_bound else 0.0)
            + (1.5 if geometry_bound else 0.0)
            + (2.0 if relation_bound else 0.0)
            + (1.5 if superlative_bound else 0.0)
        )
        best_score = max(best_score, score)
        row_details.append({
            "source_hash": stable_hash({
                "title": str(row.get("title") or ""),
                "snippet": str(row.get("snippet") or ""),
                "source": str(row.get("source") or ""),
            }),
            "oxidation_bound": oxidation_bound,
            "spin_bound": spin_bound,
            "geometry_bound": geometry_bound,
            "hyperfine_relation_bound": relation_bound,
            "superlative_bound": superlative_bound,
            "pair_bound": pair_bound,
            "direct_pair_binding": direct,
            "score": round(score, 4),
        })
    if not row_details:
        return {
            **base,
            "status": "blocked",
            "reason": "no_source_rows",
            "option_feature_hash": features["option_feature_hash"],
        }
    status = "activated" if direct_count > 0 else ("evaluated" if partial_count > 0 else "blocked")
    reason = (
        "candidate_geometry_spin_oxidation_hyperfine_relation_bound"
        if direct_count > 0
        else (
            "candidate_spin_oxidation_hyperfine_bound_missing_geometry_or_superlative"
            if partial_count > 0
            else "no_candidate_spin_oxidation_hyperfine_pair_binding"
        )
    )
    return {
        **base,
        "status": status,
        "reason": reason,
        "option_feature_hash": features["option_feature_hash"],
        "oxidation": oxidation,
        "spin_hash": stable_hash({"spin": spin}) if spin else "",
        "geometry_hash": stable_hash({"geometry_terms": geometry_terms}),
        "source_hashes": _source_hashes(rows),
        "row_count": len(row_details),
        "partial_pair_binding_row_count": partial_count,
        "direct_pair_binding_row_count": direct_count,
        "missing_geometry_row_count": missing_geometry_count,
        "missing_relation_row_count": missing_relation_count,
        "best_pair_binding_score": round(best_score, 4),
        "row_details": row_details[:5],
    }
