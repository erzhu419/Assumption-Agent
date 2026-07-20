"""P13 typed-query projection formed from the label-free P12 runtime failure."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_acquisition_v1 as utilities,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_c_confirm_runtime_v1 as p12,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_bridge_expansion_core_v1 as bridge,
)


CANDIDATE_NAME = "P13_P12_BRIDGE_SAFE_TYPED_QUERY_V1"
AUDIT_SCHEMA = "nanobeir_p13_bridge_safe_typed_queries_v1"
ROLE_PREFIXES = ("relation: ", "mechanism: ")
BRIDGE_SAFE_QUERY_CHARACTERS = (
    bridge.MAX_BRIDGE_QUERY_CHARACTERS - 1 - bridge.MAX_ANCHOR_CHARACTERS
)


class P13BridgeSafeError(RuntimeError):
    """P13 projection failed its fixed totality contract."""


def bridge_safe_query(text: object, role: str) -> str:
    if role not in ("relation", "mechanism"):
        raise P13BridgeSafeError("bridge role is outside the frozen registry")
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise P13BridgeSafeError("bridge source query is invalid")
    prefix = f"{role}: "
    normalized = " ".join(text.split())
    room = BRIDGE_SAFE_QUERY_CHARACTERS - len(prefix)
    projected = prefix + normalized[:room].rstrip()
    if (
        len(projected) > BRIDGE_SAFE_QUERY_CHARACTERS
        or not projected.removeprefix(prefix).strip()
    ):
        raise P13BridgeSafeError("bridge-safe projection drifted")
    return projected


def totalize_and_project_qwen_output(
    output: Mapping[str, Any], items: Sequence[Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    totalized, p12_audit = p12.totalize_qwen_output(output, items)
    rows = totalized.get("items")
    audit_rows = p12_audit.get("items")
    if (
        not isinstance(rows, list)
        or not isinstance(audit_rows, list)
        or len(rows) != len(items)
        or len(audit_rows) != len(items)
    ):
        raise P13BridgeSafeError("P12 totalization shape drifted")
    projected_rows: list[dict[str, Any]] = []
    projected_audit: list[dict[str, Any]] = []
    for ordinal, (row, audit_row) in enumerate(zip(rows, audit_rows)):
        expansions = row.get("expansions") if isinstance(row, Mapping) else None
        if not isinstance(expansions, list) or len(expansions) != 4:
            raise P13BridgeSafeError("totalized expansion row drifted")
        before = list(expansions)
        after = list(before)
        after[1] = bridge_safe_query(before[1], "relation")
        after[2] = bridge_safe_query(before[2], "mechanism")
        if (
            after[0] != before[0]
            or after[3] != before[3]
            or not after[1].startswith(ROLE_PREFIXES[0])
            or not after[2].startswith(ROLE_PREFIXES[1])
            or after[1] == after[2]
        ):
            raise P13BridgeSafeError("typed-query role projection drifted")
        copied = dict(row)
        copied["expansions"] = after
        copied["generation_valid"] = True
        projected_rows.append(copied)
        projected_audit.append(
            {
                "completion_sha256": audit_row.get("completion_sha256"),
                "ordinal": ordinal,
                "preprojection_expansion_sha256": [
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for value in before
                ],
                "projected_expansions": after,
                "source_generation_valid": audit_row.get(
                    "source_generation_valid"
                ),
                "totalization_used": audit_row.get("totalization_used"),
            }
        )
    projected = {"items": projected_rows, "schema": totalized.get("schema")}
    audit = utilities.self_hashed(
        {
            "bridge_query_character_cap": BRIDGE_SAFE_QUERY_CHARACTERS,
            "candidate": CANDIDATE_NAME,
            "items": projected_audit,
            "maximum_anchor_characters": bridge.MAX_ANCHOR_CHARACTERS,
            "maximum_composed_bridge_query_characters": (
                BRIDGE_SAFE_QUERY_CHARACTERS
                + 1
                + bridge.MAX_ANCHOR_CHARACTERS
            ),
            "role_prefixes": list(ROLE_PREFIXES),
            "schema": AUDIT_SCHEMA,
            "source_valid_generation_count": p12_audit[
                "source_valid_generation_count"
            ],
            "totalized_generation_count": p12_audit[
                "totalized_generation_count"
            ],
        },
        field="pack_sha256",
    )
    return projected, audit
