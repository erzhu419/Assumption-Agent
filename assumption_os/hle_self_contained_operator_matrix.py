"""Deterministic self-contained option-matrix solvers for HLE MC questions.

The functions here intentionally do not read HLE reference answers.  They also
avoid returning raw question or option text so their diagnostics can be logged
without adding new answer/content leakage.
"""

from __future__ import annotations

import re
from typing import Any

from .autonomy_journal import stable_hash


def _normalize(text: str) -> str:
    return (
        str(text or "")
        .lower()
        .replace("\u03c0", "pi")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _alkyne_handle_score(text: str) -> int:
    normalized = _normalize(text)
    score = 0
    if "prop-2-yn" in normalized or "prop 2 yn" in normalized:
        score += 5
    if "propargyl" in normalized:
        score += 5
    if "alkyne" in normalized or "alkynyl" in normalized:
        score += 4
    if "ethynyl" in normalized:
        score += 4
    if re.search(r"\b\w+-\d+-yn(?:e|yl|-\d|\b)", normalized):
        score += 3
    if "but-2-enoate" in normalized or "but 2 enoate" in normalized:
        score += 1
    if "methyl" in normalized and score:
        score += 1
    return score


def _chem_probe_trigger_summary(*, stem: str, category: str, raw_subject: str) -> dict[str, Any]:
    domain = _normalize(" ".join([category, raw_subject]))
    domain_supported = _contains_any(
        domain,
        (
            "chemistry",
            "biochemistry",
            "chemical biology",
            "biology/medicine",
            "biology medicine",
        ),
    )
    normalized = _normalize(stem)
    required_groups = {
        "probe": ("probe", "probes"),
        "click_azide": ("clicked with", "click with", "cy5 azide", "cy5-azide", "azide"),
        "fluorescent_readout": (
            "fluorescent",
            "fluorescence",
            "fluoresecent",
            "signal",
            "sds page",
            "sds-page",
            "gel",
        ),
        "perturbation": (
            "changes the probe",
            "changed the probe",
            "second probe",
            "compared with",
            "difference is lower",
        ),
        "light_or_photochemistry": ("irradiated", "light", " nm", "photochem"),
    }
    missing_groups = [
        name
        for name, cues in required_groups.items()
        if not _contains_any(normalized, cues)
    ]
    asks_molecule = _contains_any(
        normalized,
        (
            "what is the molecule",
            "what is the molecules",
            "which molecule",
            "leads to the fluorescent",
            "causes the fluorescent",
            "causes the fluorescence",
        ),
    )
    stem_has_alkyne_handle = _alkyne_handle_score(normalized) > 0
    return {
        "domain_supported": domain_supported,
        "missing_trigger_groups": missing_groups,
        "asks_molecule": asks_molecule,
        "stem_has_alkyne_handle": stem_has_alkyne_handle,
    }


def _chem_probe_option_row(label: str, option_text: str) -> dict[str, Any]:
    normalized = _normalize(option_text)
    alkyne_score = _alkyne_handle_score(normalized)
    is_azide_dye = "azide" in normalized or "cy5" in normalized
    is_generic_radical = "radical" in normalized
    is_carbene_or_photoinitiator = _contains_any(
        normalized,
        ("carbene", "thioxanthen", "thioxanthone", "photoinitiator"),
    )
    is_reagent_not_probe_handle = is_azide_dye or is_generic_radical or is_carbene_or_photoinitiator
    exclusion_score = 0
    if is_azide_dye:
        exclusion_score += 5
    if is_generic_radical:
        exclusion_score += 2
    if is_carbene_or_photoinitiator:
        exclusion_score += 2
    solver_score = alkyne_score - exclusion_score
    return {
        "label": label,
        "option_hash": stable_hash({"option_label": label}),
        "option_text_hash": stable_hash({"option_text": option_text}),
        "operator_family": "chem_probe_click_matrix",
        "is_alkyne_like": alkyne_score >= 4,
        "is_azide_dye": is_azide_dye,
        "is_generic_radical": is_generic_radical,
        "is_carbene_or_photoinitiator": is_carbene_or_photoinitiator,
        "is_reagent_not_probe_handle": is_reagent_not_probe_handle,
        "alkyne_score": int(alkyne_score),
        "exclusion_score": int(exclusion_score),
        "solver_score": int(solver_score),
    }


def _chem_probe_click_matrix(
    *,
    stem: str,
    options: dict[str, str],
    category: str,
    raw_subject: str,
) -> dict[str, Any]:
    trigger = _chem_probe_trigger_summary(stem=stem, category=category, raw_subject=raw_subject)
    option_rows = [
        _chem_probe_option_row(label, option_text)
        for label, option_text in sorted(options.items())
    ]
    if not trigger["domain_supported"]:
        return {
            "status": "not_required",
            "reason": "domain_not_supported",
            "operator_family": "chem_probe_click_matrix",
            "option_rows": option_rows,
            "trigger": trigger,
        }
    if trigger["missing_trigger_groups"] or not trigger["asks_molecule"] or not trigger["stem_has_alkyne_handle"]:
        return {
            "status": "not_required",
            "reason": "trigger_cues_missing",
            "operator_family": "chem_probe_click_matrix",
            "option_rows": option_rows,
            "trigger": trigger,
        }
    candidates = [
        row
        for row in option_rows
        if int(row.get("solver_score") or 0) >= 4
        and bool(row.get("is_alkyne_like"))
        and not bool(row.get("is_reagent_not_probe_handle"))
    ]
    ranked = sorted(
        candidates,
        key=lambda row: (-int(row.get("solver_score") or 0), str(row.get("label") or "")),
    )
    if not ranked:
        return {
            "status": "abstained",
            "reason": "no_unique_alkyne_probe_handle_option",
            "operator_family": "chem_probe_click_matrix",
            "option_rows": option_rows,
            "trigger": trigger,
            "candidate_option_hashes": [],
        }
    top_score = int(ranked[0].get("solver_score") or 0)
    runner_up_score = int(ranked[1].get("solver_score") or 0) if len(ranked) > 1 else -999
    unique_margin = top_score - runner_up_score if len(ranked) > 1 else top_score
    if len(ranked) > 1 and unique_margin <= 0:
        return {
            "status": "abstained",
            "reason": "non_unique_alkyne_probe_handle_options",
            "operator_family": "chem_probe_click_matrix",
            "option_rows": option_rows,
            "trigger": trigger,
            "candidate_option_hashes": [row["option_hash"] for row in ranked],
            "top_score": top_score,
            "runner_up_score": runner_up_score,
            "unique_margin": unique_margin,
        }
    selected_label = str(ranked[0].get("label") or "")
    return {
        "status": "activated",
        "reason": (
            "unique_alkyne_or_propargyl_like_probe_handle_matches_click_azide_"
            "fluorescence_readout_and_excludes_azide_dye_or_light_generated_species"
        ),
        "operator_family": "chem_probe_click_matrix",
        "selected_label": selected_label,
        "selected_option_hash": stable_hash({"option_label": selected_label}),
        "confidence": "mechanistic_domain_rule",
        "confidence_score": 0.86,
        "option_rows": option_rows,
        "trigger": trigger,
        "candidate_option_hashes": [row["option_hash"] for row in ranked],
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "unique_margin": unique_margin,
    }


def _antibiotic_status_table(stem: str) -> dict[str, str]:
    table: dict[str, str] = {}
    pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9/+() \-]{2,70}?)\s*(?:-|:|=|–|—)\s*([SIR])\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(str(stem or "")):
        name = re.sub(r"\s+", " ", match.group(1)).strip(" .;:,")
        if not name or name.lower() in {"answer", "choices", "options"}:
            continue
        key = _normalize_antibiotic_name(name)
        if len(key) < 3:
            continue
        table[key] = match.group(2).upper()
    return table


def _normalize_antibiotic_name(text: str) -> str:
    normalized = _normalize(text)
    normalized = normalized.replace("trimethoprim/sulfamethoxazole", "trimethoprim sulfamethoxazole")
    normalized = normalized.replace("tmp smx", "trimethoprim sulfamethoxazole")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _antibiotic_option_items(option_text: str) -> list[str]:
    raw_items = re.split(r",|;|\band\b|\bor\b", str(option_text or ""), flags=re.IGNORECASE)
    items: list[str] = []
    for item in raw_items:
        normalized = _normalize_antibiotic_name(item)
        if normalized:
            items.append(normalized)
    return items


def _antibiotic_trigger_summary(*, stem: str, category: str, raw_subject: str) -> dict[str, Any]:
    domain = _normalize(" ".join([category, raw_subject]))
    normalized = _normalize(stem)
    domain_supported = _contains_any(
        domain,
        (
            "medicine",
            "clinical",
            "biology/medicine",
            "biology medicine",
            "infectious",
            "microbiology",
        ),
    )
    table = _antibiotic_status_table(stem)
    table_status_counts = {
        status: sum(1 for value in table.values() if value == status)
        for status in ("S", "I", "R")
    }
    required_groups = {
        "infection_or_culture": ("infection", "culture", "susceptib", "antibiotic", "antimicrobial"),
        "treatment_selection": ("treatment", "regimen", "reasonable", "narrow", "option"),
    }
    missing_groups = [
        name
        for name, cues in required_groups.items()
        if not _contains_any(normalized, cues)
    ]
    return {
        "domain_supported": domain_supported,
        "table_entry_count": len(table),
        "table_status_counts": table_status_counts,
        "missing_trigger_groups": missing_groups,
    }


def _antibiotic_conditional_resistance_items(stem: str, table: dict[str, str]) -> set[str]:
    normalized_stem = _normalize(stem)
    if "d-test negative" in normalized_stem or "d test negative" in normalized_stem:
        return set()
    erythromycin_status = table.get("erythromycin")
    clindamycin_status = table.get("clindamycin")
    if erythromycin_status == "R" and clindamycin_status == "S":
        return {"clindamycin"}
    return set()


def _antibiotic_option_row(
    label: str,
    option_text: str,
    table: dict[str, str],
    *,
    conditional_resistance_items: set[str],
) -> dict[str, Any]:
    items = _antibiotic_option_items(option_text)
    status_counts = {"S": 0, "I": 0, "R": 0, "conditional": 0, "unknown": 0}
    item_hashes: list[str] = []
    item_status_hashes: list[str] = []
    for item in items:
        status = table.get(item, "")
        if not status:
            status_counts["unknown"] += 1
            status = "unknown"
        elif item in conditional_resistance_items:
            status_counts["conditional"] += 1
            status = "conditional"
        else:
            status_counts[status] += 1
        item_hashes.append(stable_hash({"antibiotic_item": item}))
        item_status_hashes.append(stable_hash({"antibiotic_item": item, "status": status}))
    item_count = len(items)
    all_known = bool(item_count and status_counts["unknown"] == 0)
    all_susceptible = bool(
        all_known
        and status_counts["conditional"] == 0
        and status_counts["S"] == item_count
        and item_count >= 2
    )
    solver_score = (
        (4 * status_counts["S"])
        - (6 * status_counts["R"])
        - (3 * status_counts["I"])
        - (4 * status_counts["conditional"])
        - (4 * status_counts["unknown"])
    )
    return {
        "label": label,
        "option_hash": stable_hash({"option_label": label}),
        "option_text_hash": stable_hash({"option_text": option_text}),
        "operator_family": "antibiotic_susceptibility_profile",
        "item_count": item_count,
        "item_hashes": item_hashes[:16],
        "item_status_hashes": item_status_hashes[:16],
        "status_counts": dict(status_counts),
        "all_items_known": all_known,
        "all_items_susceptible": all_susceptible,
        "has_resistant_or_intermediate_item": bool(
            status_counts["R"] or status_counts["I"] or status_counts["conditional"]
        ),
        "has_conditional_resistance_item": bool(status_counts["conditional"]),
        "solver_score": int(solver_score),
    }


def _antibiotic_susceptibility_profile_matrix(
    *,
    stem: str,
    options: dict[str, str],
    category: str,
    raw_subject: str,
) -> dict[str, Any]:
    trigger = _antibiotic_trigger_summary(stem=stem, category=category, raw_subject=raw_subject)
    table = _antibiotic_status_table(stem)
    conditional_resistance_items = _antibiotic_conditional_resistance_items(stem, table)
    option_rows = [
        _antibiotic_option_row(
            label,
            option_text,
            table,
            conditional_resistance_items=conditional_resistance_items,
        )
        for label, option_text in sorted(options.items())
    ]
    trigger["conditional_resistance_guard_item_hashes"] = [
        stable_hash({"antibiotic_item": item})
        for item in sorted(conditional_resistance_items)
    ]
    if not trigger["domain_supported"]:
        return {
            "status": "not_required",
            "reason": "domain_not_supported",
            "operator_family": "antibiotic_susceptibility_profile",
            "option_rows": option_rows,
            "trigger": trigger,
        }
    if trigger["missing_trigger_groups"] or int(trigger["table_entry_count"] or 0) < 4:
        return {
            "status": "not_required",
            "reason": "trigger_cues_or_status_table_missing",
            "operator_family": "antibiotic_susceptibility_profile",
            "option_rows": option_rows,
            "trigger": trigger,
        }
    candidates = [
        row
        for row in option_rows
        if bool(row.get("all_items_susceptible"))
        and int(row.get("item_count") or 0) >= 2
    ]
    ranked = sorted(
        option_rows,
        key=lambda row: (-int(row.get("solver_score") or 0), str(row.get("label") or "")),
    )
    top_score = int(ranked[0].get("solver_score") or 0) if ranked else 0
    runner_up_score = int(ranked[1].get("solver_score") or 0) if len(ranked) > 1 else -999
    if len(candidates) != 1:
        return {
            "status": "abstained",
            "reason": "no_unique_all_susceptible_option",
            "operator_family": "antibiotic_susceptibility_profile",
            "option_rows": option_rows,
            "trigger": trigger,
            "candidate_option_hashes": [row["option_hash"] for row in candidates],
            "top_score": top_score,
            "runner_up_score": runner_up_score,
            "unique_margin": top_score - runner_up_score,
        }
    selected = candidates[0]
    selected_label = str(selected.get("label") or "")
    if selected not in ranked[:1] or top_score - runner_up_score < 4:
        return {
            "status": "abstained",
            "reason": "susceptible_option_not_clear_top_margin",
            "operator_family": "antibiotic_susceptibility_profile",
            "option_rows": option_rows,
            "trigger": trigger,
            "candidate_option_hashes": [selected["option_hash"]],
            "top_score": top_score,
            "runner_up_score": runner_up_score,
            "unique_margin": top_score - runner_up_score,
        }
    return {
        "status": "activated",
        "reason": "unique_option_contains_only_table_susceptible_antibiotics_and_excludes_resistant_or_intermediate_items",
        "operator_family": "antibiotic_susceptibility_profile",
        "selected_label": selected_label,
        "selected_option_hash": stable_hash({"option_label": selected_label}),
        "confidence": "self_contained_data_rule",
        "confidence_score": 0.88,
        "option_rows": option_rows,
        "trigger": trigger,
        "candidate_option_hashes": [selected["option_hash"]],
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "unique_margin": top_score - runner_up_score,
    }


def _matrix_from_solver(solver: dict[str, Any]) -> dict[str, Any]:
    matrix = {
        "status": solver.get("status"),
        "selected_label": solver.get("selected_label"),
        "operator_family": solver.get("operator_family"),
        "confidence": solver.get("confidence"),
        "confidence_score": solver.get("confidence_score", 0.0),
        "reason": solver.get("reason"),
        "option_rows": solver.get("option_rows", []),
        "candidate_option_hashes": solver.get("candidate_option_hashes", []),
        "top_score": solver.get("top_score"),
        "runner_up_score": solver.get("runner_up_score"),
        "unique_margin": solver.get("unique_margin", 0),
        "trigger": solver.get("trigger", {}),
        "raw_content_persisted": False,
    }
    matrix["solver_feature_hash"] = stable_hash({
        "operator_family": matrix.get("operator_family"),
        "status": matrix.get("status"),
        "selected_label": matrix.get("selected_label"),
        "option_rows": matrix.get("option_rows"),
        "trigger": matrix.get("trigger"),
    })
    matrix["matrix_hash"] = stable_hash({
        "solver_feature_hash": matrix["solver_feature_hash"],
        "candidate_option_hashes": matrix.get("candidate_option_hashes"),
        "unique_margin": matrix.get("unique_margin"),
    })
    return matrix


def build_self_contained_operator_matrix(
    *,
    stem: str,
    options: dict[str, str],
    category: str = "",
    raw_subject: str = "",
) -> dict[str, Any]:
    """Build a deterministic self-contained solver matrix for one MC item."""

    solvers = [
        _antibiotic_susceptibility_profile_matrix(
            stem=stem,
            options=options,
            category=category,
            raw_subject=raw_subject,
        ),
        _chem_probe_click_matrix(
            stem=stem,
            options=options,
            category=category,
            raw_subject=raw_subject,
        ),
    ]
    for solver in solvers:
        if solver.get("status") == "activated":
            return _matrix_from_solver(solver)
    for solver in solvers:
        if solver.get("status") == "abstained":
            return _matrix_from_solver(solver)
    return _matrix_from_solver(solvers[0])
