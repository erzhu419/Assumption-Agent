"""Cumulative wall-clock evidence pack for L4 supervised autonomy.

This module intentionally distinguishes cumulative elapsed evidence from a
single uninterrupted service run.  It is valid evidence for "24h+ of supervised
wall-clock execution was observed", but it does not by itself support a
"continuous 24h daemon" claim when the component intervals have gaps.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_COMPONENTS = [
    PAPER_DIR / "l4_wallclock_24h_supervised_20260613.json",
    PAPER_DIR / "l4_wallclock_24h_completion_extension_20260614.json",
]
DEFAULT_OUT = PAPER_DIR / "l4_wallclock_cumulative_24h_20260614.json"
DEFAULT_MD_OUT = Path("reconstruction/md/l4_wallclock_cumulative_24h_20260614.md")


def build_l4_wallclock_cumulative_evidence_payload(
    *,
    root: Path,
    eval_id: str = "l4_wallclock_cumulative_24h_20260614",
    components: list[Path] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    component_paths = components or DEFAULT_COMPONENTS
    rows = [_load_component(root=root, path=path) for path in component_paths]
    intervals = [row["interval"] for row in rows if row["interval"]["start"] and row["interval"]["end"]]
    intervals_sorted = sorted(intervals, key=lambda row: row["start"])
    overlaps = _count_overlaps(intervals_sorted)
    total_seconds = round(sum(float(row["metrics"]["observed_wallclock_seconds"]) for row in rows), 4)
    total_hours = round(total_seconds / 3600.0, 6)
    min_uptime = round(min((float(row["metrics"]["observed_uptime"]) for row in rows), default=0.0), 4)
    total_cycles = sum(int(row["metrics"]["cycle_count"]) for row in rows)
    total_auto_apply = sum(int(row["metrics"]["auto_apply_count"]) for row in rows)
    total_manual_review = sum(int(row["metrics"]["manual_review_count"]) for row in rows)
    total_blocked = sum(int(row["metrics"]["blocked_count"]) for row in rows)
    total_incidents = sum(int(row["metrics"]["incident_count"]) for row in rows)
    continuous_seconds = _continuous_covered_seconds(intervals_sorted)
    component_hash = stable_hash([row["component_hash"] for row in rows])
    metrics = {
        "component_count": len(rows),
        "component_hash": component_hash,
        "observed_wallclock_seconds": total_seconds,
        "observed_wallclock_hours": total_hours,
        "observed_uptime": min_uptime,
        "cycle_count": total_cycles,
        "auto_apply_count": total_auto_apply,
        "manual_review_count": total_manual_review,
        "blocked_count": total_blocked,
        "incident_count": total_incidents,
        "graph_pollution_alert_count": sum(int(row["metrics"]["graph_pollution_alert_count"]) for row in rows),
        "forbidden_auto_apply_count": sum(int(row["metrics"]["forbidden_auto_apply_count"]) for row in rows),
        "ungated_mutation_count": sum(int(row["metrics"]["ungated_mutation_count"]) for row in rows),
        "main_graph_mutation_count": sum(int(row["metrics"]["main_graph_mutation_count"]) for row in rows),
        "min_rollback_success_rate": min(
            (float(row["metrics"]["rollback_success_rate"]) for row in rows),
            default=0.0,
        ),
        "component_pass_count": sum(1 for row in rows if row["pass"]),
        "component_graph_replayable_count": sum(1 for row in rows if row["metrics"]["graph_journal_replayable"]),
        "component_queue_replayable_count": sum(1 for row in rows if row["metrics"]["queue_journal_replayable"]),
        "component_interval_overlap_count": overlaps,
        "continuous_wallclock_seconds": round(continuous_seconds, 4),
        "continuous_wallclock_hours": round(continuous_seconds / 3600.0, 6),
        "cumulative_24h_claim_allowed": total_seconds >= 24 * 3600 and min_uptime >= 0.95,
        "continuous_24h_claim_allowed": continuous_seconds >= 24 * 3600 and min_uptime >= 0.95,
        "l4_mini_72h_claim_allowed": total_seconds >= 72 * 3600 and min_uptime >= 0.95,
        "l4a_7d_claim_allowed": total_seconds >= 7 * 24 * 3600 and min_uptime >= 0.95,
        "l4a_30d_claim_allowed": total_seconds >= 30 * 24 * 3600 and min_uptime >= 0.95,
    }
    gates = {
        "components_exist": all(row["exists"] for row in rows),
        "components_pass": metrics["component_pass_count"] == metrics["component_count"],
        "component_journals_replayable": (
            metrics["component_graph_replayable_count"] == metrics["component_count"]
            and metrics["component_queue_replayable_count"] == metrics["component_count"]
        ),
        "cumulative_24h_observed": metrics["cumulative_24h_claim_allowed"] is True,
        "forbidden_auto_apply_zero": metrics["forbidden_auto_apply_count"] == 0,
        "ungated_mutation_zero": metrics["ungated_mutation_count"] == 0,
        "main_graph_mutation_zero": metrics["main_graph_mutation_count"] == 0,
        "rollback_success": metrics["min_rollback_success_rate"] >= 1.0,
        "no_interval_overlap": metrics["component_interval_overlap_count"] == 0,
        "continuous_24h_not_fabricated": metrics["continuous_24h_claim_allowed"] is (
            metrics["continuous_wallclock_seconds"] >= 24 * 3600 and metrics["observed_uptime"] >= 0.95
        ),
        "seventy_two_hour_claim_not_fabricated": metrics["l4_mini_72h_claim_allowed"] is (
            metrics["observed_wallclock_seconds"] >= 72 * 3600 and metrics["observed_uptime"] >= 0.95
        ),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "l4_wallclock_cumulative_evidence",
        "source_md": "reconstruction/md/L4_roadmap.md",
        "performance_validation": True,
        "validation_scope": (
            "Combines completed real wall-clock supervised autonomy runs into a cumulative elapsed-time "
            "evidence pack. It allows a cumulative 24h+ claim only when component logs pass replay and "
            "safety gates. It does not allow a continuous 24h claim unless one uninterrupted interval also "
            "reaches 24h."
        ),
        "components": rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "cumulative 24h+ supervised wall-clock autonomy evidence",
        "blocked_claims": [
            claim
            for claim, allowed in {
                "continuous_24h_supervised_daemon": metrics["continuous_24h_claim_allowed"],
                "72h_wallclock_service_completed": metrics["l4_mini_72h_claim_allowed"],
                "7d_wallclock_service_completed": metrics["l4a_7d_claim_allowed"],
                "30d_wallclock_service_completed": metrics["l4a_30d_claim_allowed"],
                "unbounded_24_7_autonomous_os": False,
            }.items()
            if not allowed
        ],
        "observed_wallclock_seconds": metrics["observed_wallclock_seconds"],
        "observed_wallclock_hours": metrics["observed_wallclock_hours"],
        "observed_uptime": metrics["observed_uptime"],
        "cycle_count": metrics["cycle_count"],
        "incident_count": metrics["incident_count"],
        "rollback_success_rate": metrics["min_rollback_success_rate"],
        "manual_review_backlog_max": metrics["manual_review_count"],
        "graph_pollution_alert_count": metrics["graph_pollution_alert_count"],
        "cumulative_24h_claim_allowed": metrics["cumulative_24h_claim_allowed"],
        "continuous_24h_claim_allowed": metrics["continuous_24h_claim_allowed"],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# L4 Cumulative Wall-Clock Evidence",
        "",
        f"- pass: `{payload['pass']}`",
        f"- component count: `{m['component_count']}`",
        f"- cumulative seconds: `{m['observed_wallclock_seconds']}`",
        f"- cumulative hours: `{m['observed_wallclock_hours']}`",
        f"- continuous hours: `{m['continuous_wallclock_hours']}`",
        f"- cycles: `{m['cycle_count']}`",
        f"- auto applies: `{m['auto_apply_count']}`",
        f"- manual reviews: `{m['manual_review_count']}`",
        f"- cumulative 24h claim: `{m['cumulative_24h_claim_allowed']}`",
        f"- continuous 24h claim: `{m['continuous_24h_claim_allowed']}`",
        f"- 72h claim: `{m['l4_mini_72h_claim_allowed']}`",
        "",
        "## Claim Boundary",
        "",
        "This artifact supports cumulative 24h+ supervised wall-clock evidence. It does not claim one uninterrupted 24h daemon run.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _load_component(*, root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        return {
            "path": str(path),
            "exists": False,
            "pass": False,
            "component_hash": stable_hash(["missing", str(path)]),
            "interval": {"start": None, "end": None},
            "metrics": _empty_metrics(),
        }
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    row_metrics = {
        "observed_wallclock_seconds": float(metrics.get("observed_wallclock_seconds") or 0.0),
        "observed_uptime": float(metrics.get("observed_uptime") or 0.0),
        "cycle_count": int(metrics.get("cycle_count") or 0),
        "auto_apply_count": int(metrics.get("auto_apply_count") or 0),
        "manual_review_count": int(metrics.get("manual_review_count") or 0),
        "blocked_count": int(metrics.get("blocked_count") or 0),
        "incident_count": int(metrics.get("incident_count") or 0),
        "graph_pollution_alert_count": int(metrics.get("graph_pollution_alert_count") or 0),
        "forbidden_auto_apply_count": int(metrics.get("forbidden_auto_apply_count") or 0),
        "ungated_mutation_count": int(metrics.get("ungated_mutation_count") or 0),
        "main_graph_mutation_count": int(metrics.get("main_graph_mutation_count") or 0),
        "rollback_success_rate": float(metrics.get("rollback_success_rate") or 0.0),
        "graph_journal_replayable": bool(metrics.get("graph_journal_replayable")),
        "queue_journal_replayable": bool(metrics.get("queue_journal_replayable")),
    }
    return {
        "path": str(path),
        "exists": True,
        "pass": payload.get("pass") is True,
        "component_hash": stable_hash([str(path), payload.get("eval_id"), row_metrics]),
        "interval": {
            "start": payload.get("wallclock_start"),
            "end": payload.get("wallclock_end"),
        },
        "eval_id": payload.get("eval_id"),
        "metrics": row_metrics,
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "observed_wallclock_seconds": 0.0,
        "observed_uptime": 0.0,
        "cycle_count": 0,
        "auto_apply_count": 0,
        "manual_review_count": 0,
        "blocked_count": 0,
        "incident_count": 0,
        "graph_pollution_alert_count": 0,
        "forbidden_auto_apply_count": 0,
        "ungated_mutation_count": 0,
        "main_graph_mutation_count": 0,
        "rollback_success_rate": 0.0,
        "graph_journal_replayable": False,
        "queue_journal_replayable": False,
    }


def _count_overlaps(intervals: list[dict[str, str]]) -> int:
    count = 0
    previous_end: datetime | None = None
    for row in intervals:
        start = datetime.fromisoformat(row["start"])
        end = datetime.fromisoformat(row["end"])
        if previous_end is not None and start < previous_end:
            count += 1
        previous_end = max(previous_end, end) if previous_end is not None else end
    return count


def _continuous_covered_seconds(intervals: list[dict[str, str]]) -> float:
    longest = 0.0
    current_start: datetime | None = None
    current_end: datetime | None = None
    for row in intervals:
        start = datetime.fromisoformat(row["start"])
        end = datetime.fromisoformat(row["end"])
        if current_start is None or current_end is None:
            current_start, current_end = start, end
            continue
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            longest = max(longest, (current_end - current_start).total_seconds())
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        longest = max(longest, (current_end - current_start).total_seconds())
    return max(0.0, longest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cumulative L4 wall-clock evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="l4_wallclock_cumulative_24h_20260614")
    parser.add_argument("--component", action="append", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    components = [Path(row) for row in args.component] if args.component else None
    payload = build_l4_wallclock_cumulative_evidence_payload(root=root, eval_id=args.eval_id, components=components)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
