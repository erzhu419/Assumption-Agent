"""Unified Versioned Cost Accounting Policy (`M9.5.1` `F04` Closure).

Treats raw per-call `(prompt_tokens, cached_tokens, completion_tokens, wall_ms)` as the single authoritative
data vector and computes consistent per-call floored proxy metrics across ALL stages (`source`, `solve`,
`compile`, `aborted_runs`, and `reports`) without mixing formulas or rounding orders!
"""

from __future__ import annotations

from typing import Any, Iterable

COST_POLICY_VERSION = "hegel_cost_policy_v1_m951"


def score_single_call_usage(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    wall_ms: float = 0.0,
) -> dict[str, Any]:
    """Compute exact per-call raw & proxy metrics under BOTH standard formulas."""
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


def aggregate_call_usages(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate an iterable of raw call dicts strictly by summing per-call floored scores (`0` rounding-order drift)."""
    tot_p = tot_u = tot_c = tot_o = tot_v1 = tot_v2 = 0
    tot_wall = 0.0
    n_calls = 0
    unk_attempts = 0
    for r in calls:
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
