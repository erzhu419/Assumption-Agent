"""Unified Versioned Cost Accounting Policy (`M9.5.1` `F04` + `M9.6.1` `N01` Call-UUID Conservation Closure).

Treats raw per-call `(call_uuid, prompt_tokens, cached_tokens, completion_tokens, wall_ms)` as the single
authoritative data vector and computes consistent per-call floored proxy metrics across ALL levels
(`episode`, `event`, `world`, `cluster`, `arm`, `stage_S`, `aborted_runs`, and `reports`) — NEVER reading
legacy unversioned `token_cost_proxy` fields!
"""

from __future__ import annotations

from typing import Any, Iterable

COST_POLICY_VERSION = "hegel_cost_policy_v2_m961_call_uuid_conserved"

CONSERVED_LEDGER_FIELDS: tuple[str, ...] = (
    "call_count",
    "usage_unknown_attempt_count",
    "prompt_tokens",
    "uncached_prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "total_tokens",
    "proxy_v1_u_025c_1o",
    "proxy_v2_u_010c_4o",
)


def score_single_call_usage(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    wall_ms: float = 0.0,
) -> dict[str, Any]:
    """Compute exact per-call raw & proxy metrics under BOTH standard formulas (`v1` and `v2`)."""
    p = max(0, int(prompt_tokens or 0))
    c = max(0, min(p, int(cached_tokens or 0)))
    o = max(0, int(completion_tokens or 0))
    u = p - c
    return {
        "prompt_tokens": p,
        "uncached_prompt_tokens": u,
        "cached_tokens": c,
        "completion_tokens": o,
        "total_tokens": p + o,
        "proxy_v1_u_025c_1o": u + (c // 4) + o,
        "proxy_v2_u_010c_4o": u + (c // 10) + 4 * o,
        "wall_ms": round(float(wall_ms or 0.0), 2),
    }


def deduplicate_calls_by_uuid(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate call records by `call_uuid` (if present) and verify identical token vectors on duplicate storage."""
    seen: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for r in calls:
        cid = str(r.get("call_uuid") or "").strip()
        if not cid:
            out.append(r)
            continue
        tup = (
            int(r.get("prompt_tokens", 0) or 0),
            int(r.get("cached_tokens", 0) or 0),
            int(r.get("completion_tokens", 0) or 0),
        )
        if cid in seen:
            prev = seen[cid]
            prev_tup = (
                int(prev.get("prompt_tokens", 0) or 0),
                int(prev.get("cached_tokens", 0) or 0),
                int(prev.get("completion_tokens", 0) or 0),
            )
            if tup != prev_tup:
                raise ValueError(f"Conflicting usage vectors for duplicate call_uuid `{cid}`: {tup} != {prev_tup}")
            continue
        seen[cid] = r
        out.append(r)
    return out


def aggregate_call_usages(calls: Iterable[dict[str, Any]], *, deduplicate_uuid: bool = False) -> dict[str, Any]:
    """Aggregate an iterable of raw call dicts strictly by summing per-call floored scores (`0` rounding-order drift)."""
    seq = deduplicate_calls_by_uuid(calls) if deduplicate_uuid else list(calls)
    tot_p = tot_u = tot_c = tot_o = tot_v1 = tot_v2 = 0
    tot_wall = 0.0
    n_calls = 0
    unk_attempts = 0
    for r in seq:
        s = score_single_call_usage(
            prompt_tokens=int(r.get("prompt_tokens", 0) or 0),
            cached_tokens=int(r.get("cached_tokens", 0) or 0),
            completion_tokens=int(r.get("completion_tokens", 0) or 0),
            wall_ms=float(r.get("wall_ms", 0.0) or 0.0),
        )
        tot_p += s["prompt_tokens"]
        tot_u += s["uncached_prompt_tokens"]
        tot_c += s["cached_tokens"]
        tot_o += s["completion_tokens"]
        tot_v1 += s["proxy_v1_u_025c_1o"]
        tot_v2 += s["proxy_v2_u_010c_4o"]
        tot_wall += s["wall_ms"]
        n_calls += 1
        unk_attempts += int(r.get("usage_unknown_attempt_count", 0) or 0)
    return {
        "cost_policy_version": COST_POLICY_VERSION,
        "call_count": n_calls,
        "usage_unknown_attempt_count": unk_attempts,
        "prompt_tokens": tot_p,
        "uncached_prompt_tokens": tot_u,
        "cached_tokens": tot_c,
        "completion_tokens": tot_o,
        "total_tokens": tot_p + tot_o,
        "proxy_v1_u_025c_1o": tot_v1,
        "proxy_v2_u_010c_4o": tot_v2,
        "wall_ms": round(tot_wall, 2),
    }


def sum_aggregated_ledgers(ledgers: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum already-aggregated `CostPolicy` ledgers across disjoint partitions (`episode -> event/world/cluster -> arm`)."""
    out = {k: 0 for k in CONSERVED_LEDGER_FIELDS}
    wall = 0.0
    for led in ledgers:
        for k in CONSERVED_LEDGER_FIELDS:
            out[k] += int(led.get(k, 0) or 0)
        wall += float(led.get("wall_ms", 0.0) or 0.0)
    return {
        "cost_policy_version": COST_POLICY_VERSION,
        **out,
        "wall_ms": round(wall, 2),
    }


def assert_hierarchical_ledger_conservation(
    *,
    reference_arm_ledger: dict[str, Any],
    named_partition_ledgers: dict[str, Iterable[dict[str, Any]]],
    context_label: str = "arm",
) -> dict[str, Any]:
    """Fail-closed assertion (`M9.6.1` `N01`):
    Verifies `sum(episodes) == sum(events) == sum(worlds) == sum(clusters) == reference_arm_ledger`
    across EVERY integer token & proxy field (`v1` AND `v2` simultaneously!).
    """
    checked: dict[str, dict[str, int]] = {}
    for part_name, part_seq in named_partition_ledgers.items():
        summed = sum_aggregated_ledgers(part_seq)
        mismatches: list[str] = []
        for k in CONSERVED_LEDGER_FIELDS:
            exp_v = int(reference_arm_ledger.get(k, 0) or 0)
            got_v = int(summed[k])
            if exp_v != got_v:
                mismatches.append(f"{k}: sum({part_name})={got_v} != ref={exp_v}")
        if mismatches:
            raise AssertionError(f"CostPolicy conservation violated on `{context_label}` partition `{part_name}`: {'; '.join(mismatches)}")
        checked[part_name] = {k: int(summed[k]) for k in CONSERVED_LEDGER_FIELDS}
    return {
        "conserved": True,
        "cost_policy_version": COST_POLICY_VERSION,
        "context_label": context_label,
        "partitions_verified": sorted(checked.keys()),
        "reference_totals": {k: int(reference_arm_ledger.get(k, 0) or 0) for k in CONSERVED_LEDGER_FIELDS},
    }
