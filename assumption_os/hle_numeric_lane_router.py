"""Router wrapper for the HLE numeric-threshold lane."""

from __future__ import annotations

from typing import Any

from .autonomy_journal import stable_hash
from .hle_numeric_threshold_solver import solve_numeric_threshold_lane


def route_numeric_threshold_lane(
    *,
    stem: str,
    options: dict[str, str],
    docs_by_label: dict[str, list[dict[str, Any]]],
    category: str = "",
    raw_subject: str = "",
) -> dict[str, Any]:
    solver = solve_numeric_threshold_lane(
        stem=stem,
        options=options,
        docs_by_label=docs_by_label,
        category=category,
        raw_subject=raw_subject,
    )
    payload = {
        "status": solver.get("status"),
        "reason": solver.get("reason"),
        "selection_method": (
            "numeric_threshold_direct_witness"
            if solver.get("status") == "activated"
            else "numeric_threshold_abstain"
        ),
        "selected_label": solver.get("selected_label"),
        "selected_option_hash": solver.get("selected_option_hash"),
        "direct_high_confidence": bool(solver.get("direct_high_confidence")),
        "numeric_solver": solver,
        "raw_content_persisted": False,
    }
    payload["numeric_lane_router_hash"] = stable_hash({
        "selection_method": payload["selection_method"],
        "selected_option_hash": payload.get("selected_option_hash"),
        "solver_hash": solver.get("router_payload_hash"),
    })
    return payload
