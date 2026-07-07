"""Deterministic numeric option parsing for HLE multiple-choice items."""

from __future__ import annotations

import math
import re
from typing import Any

from .autonomy_journal import stable_hash


_UNIT_PATTERN = (
    r"(?:deg\s*[cfk]|degrees?\s*(?:celsius|fahrenheit|kelvin)|"
    r"celsius|fahrenheit|kelvin|[cfk]|"
    r"%|percent|percentage|"
    r"mmol|umol|micromolar|millimolar|mol(?:ar)?|[mun]m|mm|cm|km|m|"
    r"kg|mg|ug|microgram|g|"
    r"ms|sec(?:onds?)?|s|min(?:utes?)?|h|hours?|days?|"
    r"hz|khz|mhz|ghz|pa|kpa|mpa|atm|bar|ev|kev|mev|j|kj|mj|ph)"
)

_NUMBER_PATTERN = re.compile(
    rf"""
    (?P<ineq><=|>=|<|>|<=|>=|less\s+than|greater\s+than|at\s+least|at\s+most|no\s+less\s+than|no\s+more\s+than)?
    \s*
    (?P<number>[+-]?(?:
        (?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:e|E)\s*[+-]?\d+)? |
        (?:\d+(?:\.\d*)?|\.\d+)\s*(?:x|\*)\s*10\s*\^?\s*[+-]?\d+ |
        10\s*\^\s*[+-]?\d+
    ))
    \s*
    (?P<unit>{_UNIT_PATTERN})?
    (?![A-Za-z])
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_RANGE_PATTERNS = [
    re.compile(
        rf"\bbetween\s+(?P<lo>[+-]?\d+(?:\.\d+)?)\s+and\s+(?P<hi>[+-]?\d+(?:\.\d+)?)(?:\s*(?P<unit>{_UNIT_PATTERN}))?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<lo>[+-]?\d+(?:\.\d+)?)\s*(?:-|to)\s*(?P<hi>[+-]?\d+(?:\.\d+)?)(?:\s*(?P<unit>{_UNIT_PATTERN}))",
        flags=re.IGNORECASE,
    ),
]


def _clean_numeric_text(text: str) -> str:
    return (
        str(text or "")
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00d7", "x")
        .replace("\u00b0", " deg ")
    )


def _parse_number(raw: str) -> float | None:
    text = re.sub(r"\s+", "", str(raw or "").lower())
    text = text.replace("*10", "x10")
    sci = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))x10\^?([+-]?\d+)", text)
    if sci:
        return float(sci.group(1)) * (10 ** int(sci.group(2)))
    power = re.fullmatch(r"10\^([+-]?\d+)", text)
    if power:
        return float(10 ** int(power.group(1)))
    text = re.sub(r"e([+-]?\d+)", r"e\1", text)
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _normalize_inequality(raw: str | None) -> str | None:
    text = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    if not text:
        return None
    return {
        "<": "<",
        "<=": "<=",
        ">": ">",
        ">=": ">=",
        "less than": "<",
        "at most": "<=",
        "no more than": "<=",
        "greater than": ">",
        "at least": ">=",
        "no less than": ">=",
    }.get(text)


def _normalize_unit(raw: str | None, *, prefix_context: str = "") -> tuple[str | None, str]:
    unit = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    prefix = str(prefix_context or "").lower()
    if "ph" in unit or re.search(r"\bph\s*$", prefix):
        return "pH", "ph"
    if unit in {"c", "deg c", "degree c", "degrees c", "celsius", "degree celsius", "degrees celsius"}:
        return "degC", "temperature"
    if unit in {"f", "deg f", "degree f", "degrees f", "fahrenheit", "degree fahrenheit", "degrees fahrenheit"}:
        return "degF", "temperature"
    if unit in {"k", "deg k", "kelvin", "degree kelvin", "degrees kelvin"}:
        return "K", "temperature"
    if unit in {"%", "percent", "percentage"}:
        return "%", "percentage"
    if unit in {"m", "mol", "molar", "mol ar", "millimolar", "mmol"}:
        return unit or "M", "concentration"
    if unit in {"umol", "micromolar"}:
        return "uM", "concentration"
    if unit in {"nm", "um", "mm", "cm", "m", "km"}:
        return unit, "length"
    if unit in {"kg", "g", "mg", "ug", "microgram"}:
        return unit, "mass"
    if unit in {"ms", "s", "sec", "second", "seconds", "min", "minute", "minutes", "h", "hour", "hours", "day", "days"}:
        return unit, "time"
    if unit in {"hz", "khz", "mhz", "ghz"}:
        return unit, "frequency"
    if unit in {"pa", "kpa", "mpa", "atm", "bar"}:
        return unit, "pressure"
    if unit in {"ev", "kev", "mev", "j", "kj", "mj"}:
        return unit, "energy"
    return (unit or None), "number"


def _normalize_value(value: float, unit: str | None) -> tuple[float, str | None]:
    if unit == "degC":
        return round(value + 273.15, 8), "K"
    if unit == "degF":
        return round((value - 32.0) * 5.0 / 9.0 + 273.15, 8), "K"
    if unit == "K":
        return round(value, 8), "K"
    return round(value, 12), unit


def _match_is_formula_digit(text: str, start: int, end: int, unit: str | None) -> bool:
    if unit:
        return False
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return bool((before and before.isalpha()) or (after and after.isalpha()))


def parse_numeric_values(text: str) -> list[dict[str, Any]]:
    clean = _clean_numeric_text(text)
    values: list[dict[str, Any]] = []
    for pattern in _RANGE_PATTERNS:
        for match in pattern.finditer(clean):
            lo = _parse_number(match.group("lo"))
            hi = _parse_number(match.group("hi"))
            if lo is None or hi is None:
                continue
            unit, value_type = _normalize_unit(match.group("unit"), prefix_context=clean[: match.start()])
            norm_lo, norm_unit = _normalize_value(lo, unit)
            norm_hi, _ = _normalize_value(hi, unit)
            values.append({
                "raw": match.group(0).strip(),
                "value": min(lo, hi),
                "range_low": min(lo, hi),
                "range_high": max(lo, hi),
                "unit": unit,
                "value_type": value_type,
                "normalized_value": min(norm_lo, norm_hi),
                "normalized_range_low": min(norm_lo, norm_hi),
                "normalized_range_high": max(norm_lo, norm_hi),
                "normalized_unit": norm_unit,
                "inequality": None,
                "is_range": True,
            })
    for match in _NUMBER_PATTERN.finditer(clean):
        unit, value_type = _normalize_unit(match.group("unit"), prefix_context=clean[: match.start()])
        number_start = match.start("number")
        number_end = match.end("number")
        if _match_is_formula_digit(clean, number_start, number_end, unit):
            continue
        value = _parse_number(match.group("number"))
        if value is None:
            continue
        norm_value, norm_unit = _normalize_value(value, unit)
        values.append({
            "raw": match.group(0).strip(),
            "value": value,
            "range_low": value,
            "range_high": value,
            "unit": unit,
            "value_type": value_type,
            "normalized_value": norm_value,
            "normalized_range_low": norm_value,
            "normalized_range_high": norm_value,
            "normalized_unit": norm_unit,
            "inequality": _normalize_inequality(match.group("ineq")),
            "is_range": False,
        })
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for value in values:
        key = (
            value.get("normalized_value"),
            value.get("normalized_range_low"),
            value.get("normalized_range_high"),
            value.get("normalized_unit"),
            value.get("inequality"),
            value.get("is_range"),
        )
        if key in seen:
            continue
        seen.add(key)
        value["value_hash"] = stable_hash({
            "normalized_value": value.get("normalized_value"),
            "normalized_range_low": value.get("normalized_range_low"),
            "normalized_range_high": value.get("normalized_range_high"),
            "normalized_unit": value.get("normalized_unit"),
            "inequality": value.get("inequality"),
            "is_range": value.get("is_range"),
        })
        deduped.append(value)
    return deduped


def parse_numeric_options(options: dict[str, str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parsed_count = 0
    type_counts: dict[str, int] = {}
    for label, text in sorted((options or {}).items()):
        values = parse_numeric_values(text)
        selected = values[0] if values else {}
        if selected:
            parsed_count += 1
            value_type = str(selected.get("value_type") or "number")
            type_counts[value_type] = type_counts.get(value_type, 0) + 1
        rows.append({
            "label": str(label),
            "option_hash": stable_hash({"option_label": str(label)}),
            "option_text_hash": stable_hash({"option_text": str(text or "")}),
            "parse_success": bool(selected),
            "value": selected.get("value"),
            "range_low": selected.get("range_low"),
            "range_high": selected.get("range_high"),
            "normalized_value": selected.get("normalized_value"),
            "normalized_range_low": selected.get("normalized_range_low"),
            "normalized_range_high": selected.get("normalized_range_high"),
            "unit": selected.get("unit"),
            "normalized_unit": selected.get("normalized_unit"),
            "value_type": selected.get("value_type"),
            "inequality": selected.get("inequality"),
            "is_range": bool(selected.get("is_range")) if selected else False,
            "value_hash": selected.get("value_hash"),
            "numeric_value_count": len(values),
        })
    option_count = len(rows)
    dominant_type = ""
    if type_counts:
        dominant_type = sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    numeric_rate = round(parsed_count / max(1, option_count), 4)
    status = "activated" if option_count >= 2 and parsed_count >= max(2, option_count) else "abstained"
    reason = "all_options_numeric" if status == "activated" else "insufficient_numeric_options"
    payload = {
        "status": status,
        "reason": reason,
        "option_count": option_count,
        "numeric_option_count": parsed_count,
        "numeric_option_parse_rate": numeric_rate,
        "dominant_value_type": dominant_type,
        "value_type_counts": dict(sorted(type_counts.items())),
        "option_rows": rows,
        "raw_content_persisted": False,
    }
    payload["parse_hash"] = stable_hash({
        "numeric_option_rows": [
            {
                "label": row["label"],
                "value_hash": row.get("value_hash"),
                "normalized_unit": row.get("normalized_unit"),
                "value_type": row.get("value_type"),
            }
            for row in rows
        ],
        "dominant_value_type": dominant_type,
    })
    return payload
