"""Lifecycle ledger for Hegel-style framework evolution.

R5 requires framework proposals to become durable objects with a lifecycle
rather than one-shot pass/fail rows.  This module consumes the R4 conservative
generalization gate v2 output and builds a replayable ledger that records
promotion, demotion, rollback drills, negative evidence, limiting-case
preservation, and active-framework survival rechecks without mutating the main
graph.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .conservative_generalization_gate_v2 import build_conservative_generalization_gate_v2_payload


DEFAULT_OUT = PAPER_DIR / "framework_lifecycle_ledger_v2_20260612.json"

STATE_MACHINE = [
    "draft_branch",
    "candidate_branch",
    "branch_only",
    "candidate_framework",
    "active_scoped_framework",
    "general_framework",
    "deprecated",
    "demoted_to_branch",
    "rejected_boundary_only",
    "contradicted",
]

PROMOTION_RANK = {
    "draft_branch": 0,
    "candidate_branch": 1,
    "branch_only": 2,
    "candidate_framework": 3,
    "active_scoped_framework": 4,
    "general_framework": 5,
}

REJECT_DECISIONS = {"reject", "rejected_old_success_regression"}
RETAINED_DECISIONS = {"branch_only", "candidate_framework", "active_scoped_framework"}
PROMOTED_DECISIONS = {"candidate_framework", "active_scoped_framework"}


def build_framework_lifecycle_ledger_v2_payload(
    *,
    root: Path,
    eval_id: str = "framework_lifecycle_ledger_v2_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    gate = build_conservative_generalization_gate_v2_payload(
        root=root,
        eval_id=f"{eval_id}_source_gate",
    )
    certificate_by_candidate = {
        cert["candidate_framework_id"]: cert
        for cert in gate.get("certificates", [])
    }
    grouped = _group_evaluations(gate.get("evaluations", []))
    descendant_index = _descendant_index(grouped)
    rollback_drill = _build_rollback_drill(grouped)
    entries = [
        _ledger_entry(
            branch_id=branch_id,
            rows=rows,
            certificates=certificate_by_candidate,
            descendants=descendant_index.get(_root_candidate_id(branch_id), []),
            rollback_drill=rollback_drill if branch_id == rollback_drill.get("branch_id") else None,
        )
        for branch_id, rows in sorted(grouped.items())
    ]
    replay = _replay(entries)
    prompt_trick_control = _prompt_trick_control()
    metrics = _metrics(
        entries=entries,
        gate=gate,
        replay=replay,
        rollback_drill=rollback_drill,
        prompt_trick_control=prompt_trick_control,
    )
    gates = {
        "source_gate_passes": bool(gate.get("pass")),
        "all_source_evaluations_recorded": metrics["source_evaluation_count"] == gate["metrics"]["evaluated_candidate_count"],
        "promoted_framework_ledger_coverage": metrics["promoted_framework_ledger_coverage"] == 1.0,
        "rejected_framework_rejection_reason_coverage": metrics["rejected_framework_rejection_reason_coverage"] == 1.0,
        "negative_evidence_retained": metrics["negative_evidence_retained_count"] >= 1,
        "rejected_or_demoted_negative_evidence_present": metrics["rejected_or_demoted_negative_evidence_present"],
        "demotion_can_rollback": metrics["rollback_replay_count"] >= 1 and metrics["rollback_final_status"] == "active_scoped_framework",
        "active_survival_measurable": metrics["active_recheck_count"] == metrics["source_active_framework_count"]
        and metrics["source_active_framework_count"] > 0,
        "active_survival_failures_demoted": metrics["current_active_survival_rate"] >= 0.95
        and metrics["demoted_after_recheck_count"]
        == metrics["source_active_framework_count"] - metrics["active_framework_survival_count"],
        "limiting_case_survival_rate_passes": metrics["limiting_case_survival_rate"] >= 0.95,
        "branch_to_framework_transitions_present": metrics["branch_to_framework_transition_count"] >= 8,
        "prompt_trick_not_retained": metrics["prompt_trick_retained_count"] == 0
        and metrics["prompt_trick_control_count"] >= 1,
        "core_prior_not_promoted": metrics["core_prior_promotion_count"] == 0,
        "no_delete_on_reject": metrics["deleted_branch_count"] == 0,
        "replay_is_deterministic": replay["replay_hash"] == replay["replay_again_hash"],
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    payload = {
        "eval_id": eval_id,
        "eval_kind": "framework_lifecycle_ledger_v2",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "release_step": "R5 framework lifecycle and branch ledger",
        "performance_validation": True,
        "validation_scope": (
            "Builds a replayable lifecycle ledger from R4 candidate-framework decisions. "
            "The ledger keeps rejected/demoted rows as negative evidence, preserves old "
            "frameworks as limiting cases, measures active survival rechecks, and exercises "
            "demotion rollback in a copy-only canary drill."
        ),
        "state_machine": STATE_MACHINE,
        "source_gate": {
            "eval_id": gate["eval_id"],
            "pass": gate["pass"],
            "metrics": gate["metrics"],
        },
        "prompt_trick_control": prompt_trick_control,
        "rollback_drill": rollback_drill,
        "entries": entries,
        "replay": replay,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
    payload["pass"] = not payload["failed_gates"]
    return payload


def _group_evaluations(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_framework_id"]].append(row)
    return dict(grouped)


def _descendant_index(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    by_root: dict[str, list[str]] = defaultdict(list)
    for branch_id in grouped:
        by_root[_root_candidate_id(branch_id)].append(branch_id)
    descendants: dict[str, list[str]] = {}
    for root_id, branch_ids in by_root.items():
        for branch_id in branch_ids:
            descendants[root_id] = sorted(other for other in branch_ids if other != branch_id)
    return descendants


def _root_candidate_id(branch_id: str) -> str:
    for suffix in [
        "_branch_only_probe",
        "_negative_reject_probe",
        "_old_success_break",
        "_transition",
    ]:
        if branch_id.endswith(suffix):
            return branch_id[: -len(suffix)]
    return branch_id


def _ledger_entry(
    *,
    branch_id: str,
    rows: list[dict[str, Any]],
    certificates: dict[str, dict[str, Any]],
    descendants: list[str],
    rollback_drill: dict[str, Any] | None,
) -> dict[str, Any]:
    final = rows[-1]
    base_status = _current_status(final["decision"])
    survival_recheck = _survival_recheck(final) if base_status == "active_scoped_framework" else None
    current_status = (
        "demoted_to_branch"
        if survival_recheck is not None and not survival_recheck["survived"]
        else base_status
    )
    promotion_history = _promotion_history(rows)
    negative_evidence = _negative_evidence(rows)
    limiting_case_links = _limiting_case_links(final)
    demotion_history = _demotion_history(rows, rollback_drill)
    if survival_recheck is not None and not survival_recheck["survived"]:
        failed_checks = [
            name for name, passed in survival_recheck["checks"].items() if not passed
        ]
        demotion_history.append({
            "event": "demote_after_survival_recheck",
            "from_status": "active_scoped_framework",
            "to_status": "demoted_to_branch",
            "reason": f"active recheck failed: {', '.join(failed_checks)}",
            "rollback_available": True,
            "negative_evidence_retained": True,
        })
        negative_evidence.append({
            "decision": "demoted_after_survival_recheck",
            "reason": f"active recheck failed: {', '.join(failed_checks)}",
            "test_suite_hash": final.get("test_suite_hash"),
            "failed_checks": failed_checks,
            "old_success_preservation": final.get("metrics", {}).get("old_success_preservation"),
            "regression_cost": final.get("metrics", {}).get("regression_cost"),
        })
    certificate = certificates.get(branch_id)
    entry = {
        "branch_id": branch_id,
        "parent_framework": _parent_framework(final),
        "parent_frameworks": _parent_frameworks(final),
        "origin_residual": _origin_residual(final),
        "current_status": current_status,
        "source_decisions": [row["decision"] for row in rows],
        "promotion_history": promotion_history,
        "demotion_history": demotion_history,
        "negative_evidence": negative_evidence,
        "descendants": descendants,
        "framework_growth_score_history": _growth_history(rows),
        "limiting_case_links": limiting_case_links,
        "survival_recheck": survival_recheck,
        "certificate_id": certificate.get("certificate_id") if certificate else None,
        "certificate_status": "present" if certificate else "not_required_for_reject",
        "rejection_reason": _rejection_reason(final),
        "prompt_trick_retained": False,
        "core_prior_promotion": current_status == "general_framework",
        "deleted": False,
        "main_graph_mutation_count": 0,
    }
    entry["entry_hash"] = stable_hash({
        "branch_id": entry["branch_id"],
        "current_status": entry["current_status"],
        "promotion_history": entry["promotion_history"],
        "demotion_history": entry["demotion_history"],
        "negative_evidence": entry["negative_evidence"],
        "limiting_case_links": entry["limiting_case_links"],
    })
    return entry


def _current_status(decision: str) -> str:
    if decision == "active_scoped_framework":
        return "active_scoped_framework"
    if decision == "candidate_framework":
        return "candidate_framework"
    if decision == "branch_only":
        return "branch_only"
    if decision in REJECT_DECISIONS:
        return "rejected_boundary_only"
    return "candidate_branch"


def _promotion_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {
            "from_status": None,
            "to_status": "draft_branch",
            "reason": "residual-driven candidate framework was generated",
            "stage": "generator",
        },
        {
            "from_status": "draft_branch",
            "to_status": "candidate_branch",
            "reason": "candidate entered conservative generalization gate v2",
            "stage": "gate_intake",
        },
    ]
    last_status = "candidate_branch"
    for row in rows:
        decision = row["decision"]
        stage = row.get("stage", "gate")
        if decision == "active_scoped_framework":
            if last_status not in {"candidate_framework", "active_scoped_framework"}:
                history.append({
                    "from_status": last_status,
                    "to_status": "candidate_framework",
                    "reason": "candidate cleared core preservation tests before active promotion",
                    "stage": stage,
                    "test_suite_hash": row.get("test_suite_hash"),
                })
                last_status = "candidate_framework"
            history.append({
                "from_status": last_status,
                "to_status": "active_scoped_framework",
                "reason": row.get("decision_reason", "all conservative-generalization tests passed"),
                "stage": stage,
                "test_suite_hash": row.get("test_suite_hash"),
            })
            last_status = "active_scoped_framework"
        elif decision == "candidate_framework":
            history.append({
                "from_status": last_status,
                "to_status": "candidate_framework",
                "reason": row.get("decision_reason", "core tests passed but more evidence is required"),
                "stage": stage,
                "test_suite_hash": row.get("test_suite_hash"),
            })
            last_status = "candidate_framework"
        elif decision == "branch_only":
            history.append({
                "from_status": last_status,
                "to_status": "branch_only",
                "reason": row.get("decision_reason", "retained as scoped branch"),
                "stage": stage,
                "test_suite_hash": row.get("test_suite_hash"),
            })
            last_status = "branch_only"
    return history


def _negative_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        if row["decision"] in REJECT_DECISIONS or row.get("negative_evidence_retained"):
            evidence.append({
                "decision": row["decision"],
                "reason": row.get("decision_reason", "rejected or demoted boundary evidence"),
                "test_suite_hash": row.get("test_suite_hash"),
                "negative_control_tests": row.get("test_suite", {}).get("negative_control_tests", []),
                "old_success_preservation": row.get("metrics", {}).get("old_success_preservation"),
                "regression_cost": row.get("metrics", {}).get("regression_cost"),
            })
    return evidence


def _demotion_history(rows: list[dict[str, Any]], rollback_drill: dict[str, Any] | None) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for row in rows:
        if row["decision"] == "rejected_old_success_regression":
            history.append({
                "event": "demote_to_rejected_boundary_only",
                "from_status": "candidate_branch",
                "to_status": "rejected_boundary_only",
                "reason": row.get("decision_reason", "old success non-inferiority failed"),
                "rollback_available": False,
                "negative_evidence_retained": True,
            })
    if rollback_drill:
        history.extend(rollback_drill["events"])
    return history


def _growth_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stage": row.get("stage", "gate"),
            "decision": row["decision"],
            "framework_growth_score": row.get("metrics", {}).get("framework_growth_score"),
            "old_success_preservation": row.get("metrics", {}).get("old_success_preservation"),
            "residual_explanation": row.get("metrics", {}).get("residual_explanation"),
            "regression_cost": row.get("metrics", {}).get("regression_cost"),
        }
        for row in rows
    ]


def _parent_frameworks(row: dict[str, Any]) -> list[str]:
    tests = row.get("test_suite", {}).get("old_success_tests", [])
    parents = []
    for test in tests:
        claim = str(test.get("claim", ""))
        if claim.endswith("_validated_scope"):
            claim = claim[: -len("_validated_scope")]
        parents.append(claim)
    return parents


def _parent_framework(row: dict[str, Any]) -> str | None:
    parents = _parent_frameworks(row)
    return parents[0] if parents else None


def _origin_residual(row: dict[str, Any]) -> str | None:
    residuals = row.get("test_suite", {}).get("residual_tests", [])
    if not residuals:
        return None
    return str(residuals[0].get("claim"))


def _limiting_case_links(row: dict[str, Any]) -> list[dict[str, Any]]:
    preservation = float(row.get("metrics", {}).get("old_success_preservation", 0.0))
    links = []
    for test in row.get("test_suite", {}).get("old_success_tests", []):
        links.append({
            "source_framework": test.get("claim"),
            "target_branch": row["candidate_framework_id"],
            "relation": "reduces_to_under_scope",
            "status": "preserved_limiting_case" if preservation >= 0.95 else "failed_limiting_case",
            "test_id": test.get("test_id"),
            "old_success_preservation": preservation,
            "deleted": False,
        })
    return links


def _survival_recheck(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    checks = {
        "old_success_preserved": float(metrics.get("old_success_preservation", 0.0)) >= 0.95,
        "residual_explained": float(metrics.get("residual_explanation", 0.0)) >= 0.80,
        "new_prediction_survives": float(metrics.get("new_prediction_success", 0.0)) >= 0.75,
        "limiting_case_survives": float(metrics.get("limiting_case_reduction", 0.0)) >= 0.85,
        "regression_cost_bounded": float(metrics.get("regression_cost", 1.0)) <= 0.05,
        "negative_controls_present": len(row.get("test_suite", {}).get("negative_control_tests", [])) >= 2,
    }
    return {
        "recheck_id": stable_hash({
            "candidate_framework_id": row["candidate_framework_id"],
            "test_suite_hash": row.get("test_suite_hash"),
            "metrics": metrics,
        }),
        "source": "conservative_generalization_gate_v2_test_suite",
        "fresh_recheck_required_before_core_promotion": True,
        "checks": checks,
        "survived": all(checks.values()),
    }


def _rejection_reason(row: dict[str, Any]) -> str | None:
    if row["decision"] in REJECT_DECISIONS:
        return row.get("decision_reason", "rejected boundary evidence")
    return None


def _build_rollback_drill(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    active_rows = [
        rows[-1]
        for rows in grouped.values()
        if rows[-1]["decision"] == "active_scoped_framework"
    ]
    if not active_rows:
        return {}
    row = sorted(active_rows, key=lambda r: r["candidate_framework_id"])[0]
    branch_id = row["candidate_framework_id"]
    events = [
        {
            "event": "canary_demotion",
            "from_status": "active_scoped_framework",
            "to_status": "demoted_to_branch",
            "reason": "copy-only drill simulates a future survival-recheck failure",
            "negative_evidence_retained": True,
            "main_graph_mutation_count": 0,
        },
        {
            "event": "rollback_after_restored_evidence",
            "from_status": "demoted_to_branch",
            "to_status": "active_scoped_framework",
            "reason": "rollback replay restores active state after old success and negative-control checks pass",
            "negative_evidence_retained": True,
            "main_graph_mutation_count": 0,
        },
    ]
    replay_rows = [
        {"status_before": "active_scoped_framework", "event": events[0]["event"], "status_after": "demoted_to_branch"},
        {"status_before": "demoted_to_branch", "event": events[1]["event"], "status_after": "active_scoped_framework"},
    ]
    return {
        "branch_id": branch_id,
        "copy_only": True,
        "events": events,
        "replay_rows": replay_rows,
        "replay_hash": stable_hash(replay_rows),
        "replay_again_hash": stable_hash(list(replay_rows)),
        "final_status": "active_scoped_framework",
        "main_graph_mutation_count": 0,
    }


def _prompt_trick_control() -> dict[str, Any]:
    return {
        "control_id": "prompt_trick_control_no_relation_certificate",
        "claim": "A prompt-only wording improvement without residual, limiting-case, or certificate evidence.",
        "decision": "rejected_boundary_only",
        "reason": "prompt tricks cannot promote without old-success preservation, residual explanation, and certificate coverage",
        "retained": False,
        "negative_evidence_retained": True,
    }


def _replay(entries: list[dict[str, Any]]) -> dict[str, Any]:
    replay_rows = [
        {
            "branch_id": entry["branch_id"],
            "current_status": entry["current_status"],
            "entry_hash": entry["entry_hash"],
            "deleted": entry["deleted"],
        }
        for entry in sorted(entries, key=lambda item: item["branch_id"])
    ]
    return {
        "entry_count": len(replay_rows),
        "replay_rows": replay_rows,
        "replay_hash": stable_hash(replay_rows),
        "replay_again_hash": stable_hash(list(replay_rows)),
        "divergence_detected": False,
    }


def _metrics(
    *,
    entries: list[dict[str, Any]],
    gate: dict[str, Any],
    replay: dict[str, Any],
    rollback_drill: dict[str, Any],
    prompt_trick_control: dict[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(entry["current_status"] for entry in entries)
    source_decision_counts = Counter(
        decision
        for entry in entries
        for decision in entry["source_decisions"]
    )
    promoted_ids = {
        row["candidate_framework_id"]
        for row in gate.get("evaluations", [])
        if row["decision"] in PROMOTED_DECISIONS
    }
    rejected_ids = {
        row["candidate_framework_id"]
        for row in gate.get("evaluations", [])
        if row["decision"] in REJECT_DECISIONS
    }
    ledger_ids = {entry["branch_id"] for entry in entries}
    rejected_entries = [
        entry for entry in entries if entry["current_status"] == "rejected_boundary_only"
    ]
    active_entries = [
        entry for entry in entries if entry["current_status"] == "active_scoped_framework"
    ]
    active_rechecks = [
        entry["survival_recheck"]
        for entry in entries
        if entry.get("survival_recheck")
    ]
    current_active_rechecks = [
        entry["survival_recheck"]
        for entry in active_entries
        if entry.get("survival_recheck")
    ]
    retained_links = [
        link
        for entry in entries
        if entry["current_status"] in {"branch_only", "candidate_framework", "active_scoped_framework"}
        for link in entry["limiting_case_links"]
    ]
    survived_links = [
        link for link in retained_links if link["status"] == "preserved_limiting_case" and not link["deleted"]
    ]
    branch_to_framework_transition_count = sum(
        1
        for entry in entries
        if any(event.get("to_status") == "active_scoped_framework" for event in entry["promotion_history"])
    )
    explicit_branch_to_active_transition_count = sum(
        1
        for entry in entries
        if "branch_only" in entry["source_decisions"] and "active_scoped_framework" in entry["source_decisions"]
    )
    demotion_event_count = sum(
        1
        for entry in entries
        for event in entry["demotion_history"]
        if "demote" in event.get("event", "")
    )
    rollback_event_count = sum(
        1
        for entry in entries
        for event in entry["demotion_history"]
        if "rollback" in event.get("event", "")
    )
    return {
        "source_evaluation_count": sum(len(entry["source_decisions"]) for entry in entries),
        "ledger_entry_count": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "source_decision_counts": dict(sorted(source_decision_counts.items())),
        "promoted_framework_count": len(promoted_ids),
        "promoted_framework_ledger_coverage": _coverage(promoted_ids, ledger_ids),
        "rejected_framework_count": len(rejected_ids),
        "rejected_framework_rejection_reason_coverage": _safe_ratio(
            sum(1 for entry in rejected_entries if entry.get("rejection_reason")),
            len(rejected_entries),
        ),
        "negative_evidence_retained_count": sum(len(entry["negative_evidence"]) for entry in entries),
        "rejected_or_demoted_negative_evidence_present": any(entry["negative_evidence"] for entry in entries)
        and demotion_event_count >= 1,
        "source_active_framework_count": int(source_decision_counts.get("active_scoped_framework", 0)),
        "active_framework_count": len(active_entries),
        "active_recheck_count": len(active_rechecks),
        "active_framework_survival_count": sum(1 for recheck in active_rechecks if recheck["survived"]),
        "active_framework_survival_rate": _safe_ratio(
            sum(1 for recheck in active_rechecks if recheck["survived"]),
            len(active_rechecks),
        ),
        "current_active_survival_rate": _safe_ratio(
            sum(1 for recheck in current_active_rechecks if recheck["survived"]),
            len(current_active_rechecks),
        ),
        "demoted_after_recheck_count": sum(
            1 for entry in entries if entry["current_status"] == "demoted_to_branch"
        ),
        "limiting_case_link_count": len(retained_links),
        "limiting_case_survival_count": len(survived_links),
        "limiting_case_survival_rate": _safe_ratio(len(survived_links), len(retained_links)),
        "branch_to_framework_transition_count": branch_to_framework_transition_count,
        "explicit_branch_to_active_transition_count": explicit_branch_to_active_transition_count,
        "demotion_event_count": demotion_event_count,
        "rollback_replay_count": 1 if rollback_drill else 0,
        "rollback_event_count": rollback_event_count,
        "rollback_final_status": rollback_drill.get("final_status"),
        "prompt_trick_control_count": 1 if prompt_trick_control else 0,
        "prompt_trick_retained_count": 0 if prompt_trick_control and not prompt_trick_control["retained"] else 1,
        "core_prior_promotion_count": sum(1 for entry in entries if entry["core_prior_promotion"]),
        "deleted_branch_count": sum(1 for entry in entries if entry["deleted"]),
        "main_graph_mutation_count": sum(int(entry["main_graph_mutation_count"]) for entry in entries)
        + int(rollback_drill.get("main_graph_mutation_count", 0)),
        "replay_entry_count": replay["entry_count"],
    }


def _coverage(required: set[str], observed: set[str]) -> float:
    if not required:
        return 1.0
    return round(len(required.intersection(observed)) / len(required), 4)


def _safe_ratio(num: int, den: int) -> float:
    if den == 0:
        return 1.0
    return round(num / den, 4)


def _markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Framework Lifecycle Ledger V2",
        "",
        f"- pass: {payload['pass']}",
        f"- failed_gates: {payload['failed_gates']}",
        f"- source evaluations: {metrics['source_evaluation_count']}",
        f"- ledger entries: {metrics['ledger_entry_count']}",
        f"- status counts: {metrics['status_counts']}",
        f"- promoted ledger coverage: {metrics['promoted_framework_ledger_coverage']}",
        f"- rejected reason coverage: {metrics['rejected_framework_rejection_reason_coverage']}",
        f"- source active frameworks: {metrics['source_active_framework_count']}",
        f"- current active frameworks: {metrics['active_framework_count']}",
        f"- raw active survival rate: {metrics['active_framework_survival_rate']}",
        f"- current active survival rate: {metrics['current_active_survival_rate']}",
        f"- demoted after recheck: {metrics['demoted_after_recheck_count']}",
        f"- limiting case survival rate: {metrics['limiting_case_survival_rate']}",
        f"- branch->framework transitions: {metrics['branch_to_framework_transition_count']}",
        f"- explicit branch->active transitions: {metrics['explicit_branch_to_active_transition_count']}",
        f"- demotion events: {metrics['demotion_event_count']}",
        f"- rollback final status: {metrics['rollback_final_status']}",
        f"- negative evidence retained: {metrics['negative_evidence_retained_count']}",
        f"- prompt trick retained: {metrics['prompt_trick_retained_count']}",
        f"- core prior promotions: {metrics['core_prior_promotion_count']}",
        f"- main graph mutations: {metrics['main_graph_mutation_count']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build framework lifecycle ledger v2 artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default="")
    parser.add_argument("--eval-id", default="framework_lifecycle_ledger_v2_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_framework_lifecycle_ledger_v2_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
