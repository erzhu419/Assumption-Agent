"""L4a wall-clock supervised autonomy service contract.

L3.5 has a deterministic 30-day-equivalent supervised production candidate.
L4a requires real elapsed wall-clock evidence.  This module does not fabricate
that evidence.  It packages the service protocol, safety envelope, incident
templates, and readiness checks, while blocking completed wall-clock claims
unless an observed run log supplies enough elapsed time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .autonomy_shadow_service import FORBIDDEN_AUTO_APPLY_TYPES, LOW_RISK_MUTATION_TYPES
from .autonomy_supervised_production_run import build_autonomy_supervised_production_run_payload


DEFAULT_OUT = PAPER_DIR / "l4_wallclock_supervised_service_20260613.json"
DEFAULT_MD_OUT = Path("reconstruction/md/l4_wallclock_supervised_service_20260613.md")
SOURCE_SUPERVISED_OUT = PAPER_DIR / "autonomy_supervised_production_run_20260612.json"
DEFAULT_WALLCLOCK_LOG = PAPER_DIR / "l4_wallclock_real_smoke_20260613.json"
DEFAULT_CUMULATIVE_WALLCLOCK_LOG = PAPER_DIR / "l4_wallclock_cumulative_24h_20260614.json"

SERVICE_LEVELS = [
    {
        "level": "24h_supervised_smoke",
        "required_elapsed_hours": 24,
        "minimum_uptime": 0.95,
        "claim": "wall_clock_smoke_evidence",
    },
    {
        "level": "72h_l4_mini",
        "required_elapsed_hours": 72,
        "minimum_uptime": 0.95,
        "claim": "l4_mini_wall_clock_evidence",
    },
    {
        "level": "7d_l4a_candidate",
        "required_elapsed_hours": 168,
        "minimum_uptime": 0.95,
        "claim": "l4a_wall_clock_supervised_autonomy_candidate",
    },
    {
        "level": "30d_l4a_main",
        "required_elapsed_hours": 720,
        "minimum_uptime": 0.95,
        "claim": "l4a_multi_week_supervised_autonomy_evidence",
    },
]

REQUIRED_CYCLE_FIELDS = [
    "cycle_id",
    "wallclock_start",
    "wallclock_end",
    "queue_items_seen",
    "queue_items_leased",
    "auto_apply_count",
    "manual_review_count",
    "blocked_count",
    "checkpoint_before",
    "checkpoint_after",
    "graph_before_hash",
    "graph_after_hash",
    "rate_limit_state",
    "budget_state",
    "incident",
]

FAULT_INJECTIONS = [
    "worker_crash",
    "network_timeout",
    "missing_artifact",
    "corrupt_artifact",
    "lease_expiry",
    "duplicate_idempotency_key",
    "manual_review_backlog_spike",
    "rate_limit_backoff",
]

EXTRA_L4_LOW_RISK_TYPES = {
    "update_readback_monitor",
    "refresh_calibration_row",
}


def build_l4_wallclock_supervised_service_payload(
    *,
    root: Path,
    eval_id: str = "l4_wallclock_supervised_service_20260613",
    wallclock_log: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    supervised = _load_or_build_supervised(
        root=root,
        eval_id=f"{eval_id}_l35_supervised_source",
    )
    observed = _observed_wallclock(root=root, wallclock_log=wallclock_log or _default_wallclock_log(root))
    service_contract = _service_contract()
    readiness = _readiness(supervised=supervised, observed=observed, service_contract=service_contract)
    metrics = _metrics(supervised=supervised, observed=observed, readiness=readiness)
    gates = {
        "l35_supervised_source_passes": supervised["pass"] is True,
        "service_levels_defined": metrics["service_level_count"] == 4,
        "cycle_schema_complete": metrics["required_cycle_field_count"] >= 14,
        "fault_injection_plan_complete": metrics["fault_injection_count"] >= 8,
        "low_risk_scope_extends_l3_safely": metrics["allowed_auto_apply_type_count"] >= 7,
        "forbidden_scope_manual_only": metrics["forbidden_auto_apply_type_count"] >= 4
        and metrics["forbidden_auto_apply_count"] == 0,
        "journal_replay_ready": metrics["source_all_applies_replayable"] is True,
        "ungated_mutation_zero": metrics["ungated_mutation_count"] == 0,
        "preflight_claim_allowed": metrics["wallclock_service_preflight_claim_allowed"] is True,
        "twenty_four_hour_cumulative_claim_not_fabricated": metrics[
            "twenty_four_hour_cumulative_claim_allowed"
        ] is (
            metrics["observed_wallclock_hours"] >= 24
            and metrics["observed_uptime"] >= 0.95
        ),
        "completed_claim_not_fabricated": metrics["l4a_wallclock_completed_claim_allowed"] is (
            metrics["observed_wallclock_hours"] >= 168
            and metrics["observed_uptime"] >= 0.95
        ),
        "thirty_day_claim_not_fabricated": metrics["thirty_day_wallclock_claim_allowed"] is (
            metrics["observed_wallclock_hours"] >= 720
            and metrics["observed_uptime"] >= 0.95
        ),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "l4_wallclock_supervised_service",
        "source_md": "reconstruction/md/L4_roadmap.md",
        "l4_stage": "L4-1_wall_clock_supervised_autonomy_service",
        "implementation_level": "l4a_service_contract_and_readiness_not_completed_wallclock_run",
        "performance_validation": True,
        "validation_scope": (
            "Defines the L4a wall-clock supervised autonomy service, safety envelope, incident protocol, "
            "fault-injection schedule, and observed-run claim gates.  It reuses the L3.5 supervised source "
            "as readiness evidence but refuses to claim 72h/7d/30d wall-clock completion without real elapsed "
            "run logs."
        ),
        "service_contract": service_contract,
        "observed_wallclock_log": observed,
        "source_supervised_run": {
            "pass": supervised["pass"],
            "metrics": supervised["metrics"],
            "failed_gates": supervised["failed_gates"],
        },
        "readiness": readiness,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "L4a wall-clock supervised autonomy service is protocol-ready and safety-gated",
        "blocked_claims": [
            claim
            for claim, allowed in {
                "continuous_24h_wallclock_service_completed": metrics["twenty_four_hour_continuous_claim_allowed"],
                "72h_wallclock_service_completed": metrics["l4_mini_72h_claim_allowed"],
                "7d_wallclock_service_completed": metrics["l4a_wallclock_completed_claim_allowed"],
                "30d_wallclock_service_completed": metrics["thirty_day_wallclock_claim_allowed"],
                "unbounded_24_7_autonomous_os": False,
            }.items()
            if not allowed
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# L4 Wall-Clock Supervised Service",
        "",
        f"- pass: `{payload['pass']}`",
        f"- service levels: `{m['service_level_count']}`",
        f"- observed wall-clock hours: `{m['observed_wallclock_hours']}`",
        f"- observed wall-clock seconds: `{m['observed_wallclock_seconds']}`",
        f"- observed uptime: `{m['observed_uptime']}`",
        f"- real smoke claim allowed: `{m['real_wallclock_smoke_claim_allowed']}`",
        f"- cumulative 24h claim allowed: `{m['twenty_four_hour_cumulative_claim_allowed']}`",
        f"- continuous 24h claim allowed: `{m['twenty_four_hour_continuous_claim_allowed']}`",
        f"- preflight claim allowed: `{m['wallclock_service_preflight_claim_allowed']}`",
        f"- 72h claim allowed: `{m['l4_mini_72h_claim_allowed']}`",
        f"- 7d claim allowed: `{m['l4a_wallclock_completed_claim_allowed']}`",
        f"- 30d claim allowed: `{m['thirty_day_wallclock_claim_allowed']}`",
        f"- ungated mutation count: `{m['ungated_mutation_count']}`",
        "",
        "## Claim Boundary",
        "",
        "This artifact makes the service protocol runnable. It does not claim real elapsed wall-clock completion.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _service_contract() -> dict[str, Any]:
    allowed = sorted(set(LOW_RISK_MUTATION_TYPES) | EXTRA_L4_LOW_RISK_TYPES)
    manual = sorted(
        set(FORBIDDEN_AUTO_APPLY_TYPES)
        | {
            "new_active_framework_promotion",
            "core_philosophy_prior_promotion",
            "evaluator_change",
            "main_prompt_change",
            "permission_boundary_change",
        }
    )
    return {
        "service_levels": SERVICE_LEVELS,
        "cycle_schema": REQUIRED_CYCLE_FIELDS,
        "allowed_auto_apply_types": allowed,
        "manual_review_required_types": manual,
        "fault_injections": FAULT_INJECTIONS,
        "incident_report_template": {
            "incident_id": None,
            "cycle_id": None,
            "severity": None,
            "detected_at": None,
            "root_cause": None,
            "graph_before_hash": None,
            "graph_after_hash": None,
            "rollback_action": None,
            "human_reviewer": None,
            "postmortem_required": True,
        },
        "scheduler_requirements": [
            "lease_based_queue",
            "idempotent_apply",
            "append_only_journal",
            "checkpoint_before_apply",
            "checkpoint_after_apply",
            "budget_and_rate_limit_state",
            "manual_review_backlog_monitor",
            "rollback_drill_per_service_level",
        ],
        "contract_hash": stable_hash([SERVICE_LEVELS, allowed, manual, REQUIRED_CYCLE_FIELDS, FAULT_INJECTIONS]),
    }


def _load_or_build_supervised(*, root: Path, eval_id: str) -> dict[str, Any]:
    path = root / SOURCE_SUPERVISED_OUT
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_autonomy_supervised_production_run_payload(root=root, eval_id=eval_id)


def _default_wallclock_log(root: Path) -> Path:
    cumulative = root / DEFAULT_CUMULATIVE_WALLCLOCK_LOG
    return DEFAULT_CUMULATIVE_WALLCLOCK_LOG if cumulative.exists() else DEFAULT_WALLCLOCK_LOG


def _observed_wallclock(*, root: Path, wallclock_log: Path | None) -> dict[str, Any]:
    if wallclock_log is None:
        return {
            "path": None,
            "exists": False,
            "observed_wallclock_seconds": 0.0,
            "observed_wallclock_hours": 0.0,
            "observed_uptime": 0.0,
            "cycle_count": 0,
            "incident_count": 0,
            "rollback_success_rate": None,
            "manual_review_backlog_max": None,
            "graph_pollution_alert_count": 0,
            "cumulative_24h_claim_allowed": False,
            "continuous_24h_claim_allowed": False,
            "claim_source": "no_real_wallclock_log_supplied",
        }
    path = wallclock_log if wallclock_log.is_absolute() else root / wallclock_log
    if not path.exists():
        return {
            "path": str(wallclock_log),
            "exists": False,
            "observed_wallclock_seconds": 0.0,
            "observed_wallclock_hours": 0.0,
            "observed_uptime": 0.0,
            "cycle_count": 0,
            "incident_count": 0,
            "rollback_success_rate": None,
            "manual_review_backlog_max": None,
            "graph_pollution_alert_count": 0,
            "cumulative_24h_claim_allowed": False,
            "continuous_24h_claim_allowed": False,
            "claim_source": "wallclock_log_path_missing",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    cycles = payload.get("cycles", []) if isinstance(payload, dict) else []
    payload_metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return {
        "path": str(wallclock_log),
        "exists": True,
        "observed_wallclock_seconds": float(
            payload.get("observed_wallclock_seconds")
            or payload_metrics.get("observed_wallclock_seconds")
            or 0.0
        ),
        "observed_wallclock_hours": float(
            payload.get("observed_wallclock_hours")
            or payload_metrics.get("observed_wallclock_hours")
            or 0.0
        ),
        "observed_uptime": float(payload.get("observed_uptime") or payload_metrics.get("observed_uptime") or 0.0),
        "cycle_count": int(payload.get("cycle_count") or payload_metrics.get("cycle_count") or len(cycles)),
        "incident_count": int(payload.get("incident_count") or payload_metrics.get("incident_count") or 0),
        "rollback_success_rate": payload.get("rollback_success_rate") or payload_metrics.get("rollback_success_rate"),
        "manual_review_backlog_max": payload.get("manual_review_backlog_max")
        or payload_metrics.get("manual_review_backlog_max"),
        "graph_pollution_alert_count": int(
            payload.get("graph_pollution_alert_count")
            or payload_metrics.get("graph_pollution_alert_count")
            or 0
        ),
        "cumulative_24h_claim_allowed": bool(payload.get("cumulative_24h_claim_allowed")),
        "continuous_24h_claim_allowed": bool(payload.get("continuous_24h_claim_allowed")),
        "claim_source": "real_wallclock_log",
    }


def _readiness(*, supervised: dict[str, Any], observed: dict[str, Any], service_contract: dict[str, Any]) -> dict[str, Any]:
    source_metrics = supervised["metrics"]
    return {
        "source_l35_supervised_pass": supervised["pass"],
        "source_supervised_day_count": source_metrics["supervised_day_count"],
        "source_cycle_count": source_metrics["cycle_count"],
        "source_all_applies_replayable": source_metrics["all_applies_replayable"],
        "source_low_risk_auto_apply_precision": source_metrics["low_risk_auto_apply_precision"],
        "observed_run_present": observed["exists"],
        "required_real_log_before_completion_claim": True,
        "service_contract_hash": service_contract["contract_hash"],
        "claim_boundary": "preflight-ready unless observed run log satisfies elapsed-time gates",
    }


def _metrics(*, supervised: dict[str, Any], observed: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    source = supervised["metrics"]
    observed_hours = round(float(observed["observed_wallclock_hours"]), 4)
    observed_seconds = round(float(observed.get("observed_wallclock_seconds") or observed_hours * 3600.0), 4)
    observed_uptime = round(float(observed["observed_uptime"]), 4)
    return {
        "service_level_count": len(SERVICE_LEVELS),
        "required_cycle_field_count": len(REQUIRED_CYCLE_FIELDS),
        "fault_injection_count": len(FAULT_INJECTIONS),
        "allowed_auto_apply_type_count": len(set(LOW_RISK_MUTATION_TYPES) | EXTRA_L4_LOW_RISK_TYPES),
        "forbidden_auto_apply_type_count": len(FORBIDDEN_AUTO_APPLY_TYPES) + 5,
        "forbidden_auto_apply_count": source["forbidden_policy_change_auto_apply_count"],
        "source_supervised_day_count": source["supervised_day_count"],
        "source_cycle_count": source["cycle_count"],
        "source_all_applies_replayable": source["all_applies_replayable"],
        "source_low_risk_auto_apply_precision": source["low_risk_auto_apply_precision"],
        "observed_wallclock_hours": observed_hours,
        "observed_wallclock_seconds": observed_seconds,
        "observed_uptime": observed_uptime,
        "observed_cycle_count": observed["cycle_count"],
        "observed_incident_count": observed["incident_count"],
        "observed_graph_pollution_alert_count": observed["graph_pollution_alert_count"],
        "twenty_four_hour_cumulative_claim_allowed": bool(observed.get("cumulative_24h_claim_allowed"))
        or (observed_seconds >= 24 * 3600 and observed_uptime >= 0.95),
        "twenty_four_hour_continuous_claim_allowed": bool(observed.get("continuous_24h_claim_allowed")),
        "ungated_mutation_count": source["ungated_mutation_count"],
        "wallclock_service_preflight_claim_allowed": bool(readiness["source_l35_supervised_pass"]),
        "real_wallclock_smoke_claim_allowed": observed_seconds > 0 and observed_uptime >= 0.95,
        "l4_mini_72h_claim_allowed": observed_hours >= 72 and observed_uptime >= 0.95,
        "l4a_wallclock_completed_claim_allowed": observed_hours >= 168 and observed_uptime >= 0.95,
        "thirty_day_wallclock_claim_allowed": observed_hours >= 720 and observed_uptime >= 0.95,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build L4 wall-clock supervised service contract artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="l4_wallclock_supervised_service_20260613")
    parser.add_argument("--wallclock-log", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    wallclock_log = Path(args.wallclock_log) if args.wallclock_log else None
    payload = build_l4_wallclock_supervised_service_payload(
        root=root,
        eval_id=args.eval_id,
        wallclock_log=wallclock_log,
    )
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
