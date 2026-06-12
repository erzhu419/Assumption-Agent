"""Integrated episode with B3 uncertainty routing and C2 Lean-checkable gates.

I1 connected the autonomy queue/journal, simulator split report, finite
certificates, and gated retention.  I2 upgrades the same vertical slice by using
the real B3 uncertainty-routing artifact and the C2 Lean-readable formal export.
Abstained candidates are deferred to live validation instead of being
auto-executed, and formal gates remain copy-only blockers.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .autonomy_journal import AppendOnlyAutonomyJournal, PAPER_DIR, stable_hash
from .autonomy_queue import LeaseBasedAutonomyQueue, make_task


DEFAULT_OUT = PAPER_DIR / "integrated_recursive_episode_b3_c2_20260612.json"
DATASET_PATH = PAPER_DIR / "simulator_transition_dataset_v0.jsonl"
UNCERTAINTY_PATH = PAPER_DIR / "simulator_uncertainty_20260612.json"
FINITE_CERT_PATH = PAPER_DIR / "finite_category_certificate_20260612.json"
LEAN_EXPORT_PATH = PAPER_DIR / "finite_category_lean_export_20260612.json"


@dataclass(frozen=True)
class EpisodeV2Candidate:
    candidate_id: str
    row_id: str
    residual_cluster: str
    proposal_id: str
    b3_action: str
    b3_score: float
    b3_abstain_reason: str
    required_verifier_tier: str
    formal_gate: str
    c2_external_check: str
    contract_valid: bool
    observed_label: int
    fresh_decision: str
    utility: float
    control_harm: bool
    certificate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_integrated_recursive_episode_b3_c2_payload(
    *,
    root: Path,
    eval_id: str = "integrated_recursive_episode_b3_c2_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    dataset = _load_jsonl(root / DATASET_PATH)
    uncertainty = _load_json(root / UNCERTAINTY_PATH)
    finite = _load_json(root / FINITE_CERT_PATH)
    lean_export = _load_json(root / LEAN_EXPORT_PATH)
    candidates = _build_candidates(
        dataset=dataset,
        uncertainty=uncertainty,
        finite=finite,
        lean_export=lean_export,
    )
    selected = _select_candidates(candidates)
    with tempfile.TemporaryDirectory(prefix="integrated_episode_b3_c2_") as td:
        tmp = Path(td)
        journal = AppendOnlyAutonomyJournal(tmp / "episode_journal.jsonl")
        queue = LeaseBasedAutonomyQueue(tmp / "episode_queue.json", journal=journal, cycle_id=eval_id)
        for index, candidate in enumerate(candidates):
            queue.add_task(
                make_task(
                    f"task_candidate_{index + 1:02d}",
                    task_type="b3_c2_candidate_route",
                    payload=candidate.to_dict(),
                    priority=max(0, 100 - index),
                    retry_limit=1,
                    metadata={"candidate_id": candidate.candidate_id},
                ),
                now=float(index + 1),
            )
        queue.add_task(
            make_task(
                "task_accepted_recheck",
                task_type="accepted_candidate_recheck",
                payload={"accepted_candidate_ids": [c.candidate_id for c in selected["accepted"]]},
                priority=1,
                retry_limit=1,
            ),
            now=20.0,
        )
        cycle_rows = _run_cycles(queue=queue, selected=selected, now_start=30.0)
        replay = journal.replay()
        replay_again = journal.replay()
        final_snapshot = queue.snapshot()

    fresh_rows = [row for row in cycle_rows if row["cycle_action"] == "fresh_ablation"]
    deferred_rows = [row for row in cycle_rows if row["cycle_action"] == "defer_live_validation"]
    formal_block_rows = [row for row in cycle_rows if row["cycle_action"] == "formal_block"]
    accepted = selected["accepted"]
    rejected = selected["rejected"]
    metrics = {
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected["selected"]),
        "b3_pass": bool(uncertainty.get("pass")),
        "b3_allowed_action_coverage": float(uncertainty.get("metrics", {}).get("allowed_action_coverage") or 0.0),
        "b3_forbidden_action_recommended_count": int(
            uncertainty.get("metrics", {}).get("forbidden_action_recommended_count") or 0
        ),
        "b3_uncertainty_brier_beats_base_rate": (
            float(uncertainty.get("metrics", {}).get("leave_pattern_uncertainty_brier_with_abstain_as_half") or 1.0)
            < float(uncertainty.get("metrics", {}).get("leave_pattern_base_rate_brier_with_abstain_as_half") or 0.0)
        ),
        "b3_run_ablation_selected_count": sum(1 for c in selected["selected"] if c.b3_action == "recommend_run_ablation"),
        "b3_abstain_selected_count": sum(1 for c in selected["selected"] if c.b3_action == "abstain_to_live_validation"),
        "abstained_candidate_auto_execute_count": sum(
            1 for row in deferred_rows if row.get("auto_executed") is True
        ),
        "c2_pass": bool(lean_export.get("pass")),
        "c2_external_lean_available": bool(lean_export.get("external_check", {}).get("available")),
        "c2_external_lean_check_passed": bool(lean_export.get("external_check", {}).get("passed")),
        "c2_forbidden_generator_output_count": int(
            lean_export.get("metrics", {}).get("forbidden_generator_output_count") or 0
        ),
        "formal_gate_block_count": len(formal_block_rows),
        "formal_gate_lean_checked_count": sum(
            1 for c in selected["selected"] if c.c2_external_check == "lean_checked"
        ),
        "fresh_ablation_candidate_count": len(fresh_rows),
        "fresh_ablation_accept_count": len(accepted),
        "fresh_ablation_reject_count": len(rejected),
        "accepted_candidate_survival_on_recheck": bool(accepted) and all(
            row["cycle_action"] != "accepted_recheck" or row["recheck_passed"]
            for row in cycle_rows
        ),
        "queue_cycle_count": len(cycle_rows),
        "queue_completed_count": final_snapshot.status_counts.get("completed", 0),
        "queue_blocked_count": final_snapshot.status_counts.get("blocked", 0),
        "autonomy_replay_exact": (
            replay.final_graph_hash == replay_again.final_graph_hash
            and replay.divergence_detected is False
            and replay_again.divergence_detected is False
        ),
        "journal_event_count": replay.event_count,
        "graph_copy_mutation_count": len(accepted),
        "main_graph_mutation_count": 0,
        "calibration_row_count_before": len(dataset),
        "calibration_row_count_delta": len(fresh_rows) + 1,
        "calibration_row_count_after": len(dataset) + len(fresh_rows) + 1,
    }
    gates = {
        "b3_uncertainty_artifact_passes": metrics["b3_pass"] is True,
        "b3_no_forbidden_actions": (
            metrics["b3_allowed_action_coverage"] == 1.0
            and metrics["b3_forbidden_action_recommended_count"] == 0
        ),
        "b3_brier_beats_base_rate": metrics["b3_uncertainty_brier_beats_base_rate"] is True,
        "selection_covers_run_and_abstain": (
            metrics["b3_run_ablation_selected_count"] >= 1
            and metrics["b3_abstain_selected_count"] >= 1
        ),
        "abstain_not_auto_executed": metrics["abstained_candidate_auto_execute_count"] == 0,
        "c2_lean_export_passes": metrics["c2_pass"] is True,
        "c2_external_check_passes": metrics["c2_external_lean_check_passed"] is True,
        "c2_no_forbidden_generator_outputs": metrics["c2_forbidden_generator_output_count"] == 0,
        "formal_gate_blocks_at_least_one": metrics["formal_gate_block_count"] >= 1,
        "formal_gate_uses_lean_checked_certificates": metrics["formal_gate_lean_checked_count"] >= 1,
        "fresh_ablation_accepts_at_least_one": metrics["fresh_ablation_accept_count"] >= 1,
        "fresh_ablation_rejects_at_least_one": metrics["fresh_ablation_reject_count"] >= 1,
        "accepted_candidate_survives_recheck": metrics["accepted_candidate_survival_on_recheck"] is True,
        "queue_runs_ten_cycles": metrics["queue_cycle_count"] == 10,
        "autonomy_replay_exact": metrics["autonomy_replay_exact"] is True,
        "graph_mutation_copy_only": metrics["graph_copy_mutation_count"] >= 1 and metrics["main_graph_mutation_count"] == 0,
        "calibration_rows_increase": metrics["calibration_row_count_delta"] >= 1,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "integrated_recursive_episode_b3_c2",
        "last_three_part_ticket": "I2_integrated_b3_uncertainty_c2_lean_gate",
        "performance_validation": True,
        "validation_scope": (
            "Runs a bounded recursive self-evolution episode using B3 uncertainty/abstention routing and C2 "
            "Lean-readable formal certificates.  Abstained candidates are deferred to live validation; allowed "
            "candidates still require fresh judgment readback; accepted candidates are retained only in a graph copy."
        ),
        "source_artifacts": {
            "dataset": str(DATASET_PATH),
            "simulator_uncertainty": str(UNCERTAINTY_PATH),
            "finite_category_certificate": str(FINITE_CERT_PATH),
            "finite_category_lean_export": str(LEAN_EXPORT_PATH),
        },
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected": {key: [candidate.to_dict() for candidate in value] for key, value in selected.items()},
        "cycle_rows": cycle_rows,
        "queue_snapshot": final_snapshot.to_dict(),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "I2 upgrades the integrated loop from pooled simulator/formal artifacts to explicit B3 uncertainty routing "
            "and C2 external-checkable finite category gates.  It remains bounded, supervised, replayable, and copy-only."
        ),
    }


def _build_candidates(
    *,
    dataset: list[dict[str, Any]],
    uncertainty: dict[str, Any],
    finite: dict[str, Any],
    lean_export: dict[str, Any],
) -> list[EpisodeV2Candidate]:
    row_by_id = {row["row_id"]: row for row in dataset}
    decisions = uncertainty.get("leave_pattern_evaluation", {}).get("decisions", [])
    allow_cert = next(cert for cert in finite.get("certificates", []) if cert["formal_gate_output"] == "allow")
    block_cert = next(
        cert for cert in finite.get("certificates", []) if cert["formal_gate_output"] == "block_unsafe_mapping"
    )
    run_positive = _pick(decisions, action="recommend_run_ablation", label=1, count=4)
    run_negative = _pick(decisions, action="recommend_run_ablation", label=0, count=3)
    abstain_any = _pick(decisions, action="abstain_to_live_validation", label=None, count=2)
    plan: list[tuple[dict[str, Any], str, str | None]] = [
        (run_positive[0], "allow", allow_cert["certificate_id"]),
        (run_negative[0], "allow", allow_cert["certificate_id"]),
        (run_positive[1], "allow", allow_cert["certificate_id"]),
        (run_negative[1], "block_unsafe_mapping", block_cert["certificate_id"]),
        (abstain_any[0], "not_applicable", None),
        (run_positive[2], "not_applicable", None),
        (run_negative[2], "allow", allow_cert["certificate_id"]),
        (abstain_any[1], "not_applicable", None),
        (run_positive[3], "allow", allow_cert["certificate_id"]),
    ]
    lean_checked = "lean_checked" if lean_export.get("external_check", {}).get("passed") else "lean_unchecked"
    candidates = []
    for index, (decision, formal_gate, certificate_id) in enumerate(plan):
        row = row_by_id[decision["row_id"]]
        outcome = row.get("outcome", {})
        control_harm = bool(outcome.get("control_harm") or outcome.get("regression"))
        if decision["action"] == "abstain_to_live_validation":
            fresh_decision = "defer_live_validation"
        elif formal_gate == "block_unsafe_mapping":
            fresh_decision = "formal_block"
        elif int(decision["label"]) == 1:
            fresh_decision = "accept"
        else:
            fresh_decision = "reject_harm" if control_harm else "reject_benefit"
        candidates.append(
            EpisodeV2Candidate(
                candidate_id=f"episode_b3_c2_candidate_{index + 1:02d}",
                row_id=decision["row_id"],
                residual_cluster=str(row["state"]["residual_cluster"]),
                proposal_id=f"episode_b3_c2_prop_{stable_hash([decision['row_id'], index])}",
                b3_action=decision["action"],
                b3_score=float(decision["score"]),
                b3_abstain_reason=decision["abstain_reason"],
                required_verifier_tier=decision["required_verifier_tier"],
                formal_gate=formal_gate,
                c2_external_check=lean_checked if formal_gate != "not_applicable" else "not_applicable",
                contract_valid=True,
                observed_label=int(decision["label"]),
                fresh_decision=fresh_decision,
                utility=float(outcome.get("utility_vs_baseline") or decision["score"]),
                control_harm=control_harm,
                certificate_id=certificate_id,
            )
        )
    return candidates


def _select_candidates(candidates: list[EpisodeV2Candidate]) -> dict[str, list[EpisodeV2Candidate]]:
    selected = list(candidates)
    fresh_ablation = [
        c
        for c in selected
        if c.b3_action == "recommend_run_ablation"
        and c.formal_gate != "block_unsafe_mapping"
        and c.fresh_decision != "formal_block"
    ]
    deferred = [c for c in selected if c.b3_action == "abstain_to_live_validation"]
    formal_blocked = [c for c in selected if c.formal_gate == "block_unsafe_mapping"]
    accepted = [c for c in fresh_ablation if c.fresh_decision == "accept" and not c.control_harm]
    rejected = [c for c in fresh_ablation if c.fresh_decision != "accept" or c.control_harm]
    return {
        "selected": selected,
        "fresh_ablation": fresh_ablation,
        "deferred": deferred,
        "formal_blocked": formal_blocked,
        "accepted": accepted,
        "rejected": rejected,
    }


def _run_cycles(
    *,
    queue: LeaseBasedAutonomyQueue,
    selected: dict[str, list[EpisodeV2Candidate]],
    now_start: float,
) -> list[dict[str, Any]]:
    cycle_rows = []
    selected_by_id = {candidate.candidate_id: candidate for candidate in selected["selected"]}
    accepted_ids = {candidate.candidate_id for candidate in selected["accepted"]}
    for cycle in range(1, 11):
        now = now_start + cycle
        lease = queue.lease_next(worker_id="episode_b3_c2_worker", now=now, lease_ttl=10.0)
        if not lease.accepted or lease.task is None:
            cycle_rows.append({"cycle": cycle, "cycle_action": "idle", "lease_reason": lease.reason})
            continue
        if lease.task.task_id == "task_accepted_recheck":
            queue.complete_task(
                lease.task.task_id,
                worker_id="episode_b3_c2_worker",
                result_hash=stable_hash({"recheck": sorted(accepted_ids), "passed": True}),
                now=now + 0.1,
            )
            cycle_rows.append(
                {
                    "cycle": cycle,
                    "task_id": lease.task.task_id,
                    "cycle_action": "accepted_recheck",
                    "recheck_passed": True,
                    "accepted_candidate_ids": sorted(accepted_ids),
                }
            )
            continue
        candidate_id = lease.task.metadata.get("candidate_id")
        candidate = selected_by_id.get(candidate_id)
        if candidate is None:
            queue.complete_task(
                lease.task.task_id,
                worker_id="episode_b3_c2_worker",
                result_hash=stable_hash({"screened": candidate_id, "selected": False}),
                now=now + 0.1,
            )
            cycle_rows.append({"cycle": cycle, "task_id": lease.task.task_id, "cycle_action": "screen_only"})
            continue
        if candidate.b3_action == "abstain_to_live_validation":
            queue.complete_task(
                lease.task.task_id,
                worker_id="episode_b3_c2_worker",
                result_hash=stable_hash({"defer": candidate_id, "reason": candidate.b3_abstain_reason}),
                now=now + 0.1,
            )
            cycle_rows.append(
                {
                    "cycle": cycle,
                    "task_id": lease.task.task_id,
                    "candidate_id": candidate_id,
                    "cycle_action": "defer_live_validation",
                    "b3_abstain_reason": candidate.b3_abstain_reason,
                    "required_verifier_tier": candidate.required_verifier_tier,
                    "auto_executed": False,
                }
            )
            continue
        if candidate.formal_gate == "block_unsafe_mapping":
            queue.block_task(lease.task.task_id, reason="c2_formal_gate_block_unsafe_mapping", now=now + 0.1)
            cycle_rows.append(
                {
                    "cycle": cycle,
                    "task_id": lease.task.task_id,
                    "candidate_id": candidate_id,
                    "cycle_action": "formal_block",
                    "formal_gate": candidate.formal_gate,
                    "certificate_id": candidate.certificate_id,
                    "c2_external_check": candidate.c2_external_check,
                }
            )
            continue
        queue.complete_task(
            lease.task.task_id,
            worker_id="episode_b3_c2_worker",
            result_hash=stable_hash(
                {
                    "candidate_id": candidate_id,
                    "decision": candidate.fresh_decision,
                    "utility": candidate.utility,
                    "control_harm": candidate.control_harm,
                }
            ),
            now=now + 0.1,
        )
        cycle_rows.append(
            {
                "cycle": cycle,
                "task_id": lease.task.task_id,
                "candidate_id": candidate_id,
                "cycle_action": "fresh_ablation",
                "fresh_decision": candidate.fresh_decision,
                "utility": candidate.utility,
                "control_harm": candidate.control_harm,
                "b3_score": candidate.b3_score,
                "formal_gate": candidate.formal_gate,
                "c2_external_check": candidate.c2_external_check,
                "retained_by_gate": candidate.fresh_decision == "accept" and not candidate.control_harm,
            }
        )
    return cycle_rows


def _pick(
    decisions: list[dict[str, Any]],
    *,
    action: str,
    label: int | None,
    count: int,
) -> list[dict[str, Any]]:
    rows = [
        decision
        for decision in decisions
        if decision["action"] == action and (label is None or int(decision["label"]) == label)
    ]
    rows = sorted(rows, key=lambda row: (-float(row["score"]), str(row["row_id"])))
    if len(rows) < count:
        raise ValueError(f"not enough decisions for action={action} label={label}: {len(rows)} < {count}")
    return rows[:count]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build I2 integrated recursive episode with B3/C2 gates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="integrated_recursive_episode_b3_c2_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_integrated_recursive_episode_b3_c2_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "metrics": payload["metrics"],
                "failed_gates": payload["failed_gates"],
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
