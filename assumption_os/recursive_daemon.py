"""Bounded daemon driver for recursive assumption runs.

The daemon closes the final operational gap: it can plan or execute frontier
`command_hint`s, ingest judgment bundles, resume the recursive tree, and
optionally apply accepted candidates.  It is intentionally bounded and dry-run
by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .candidate_acceptance import apply_accepted_candidates
from .graph_memory import JsonlGraphStore
from .manifest_logger import make_component_manifest
from .queue_artifact_eval import build_queue_artifact_eval_payload, judgment_sets_from_artifact_eval
from .recursive_executor import JudgmentSet, build_recursive_execution_payload, load_judgment_sets
from .schema import ResidualType, TrialStatus, stable_id


def build_recursive_daemon_payload(
    *,
    root: Path,
    graph_dir: Path,
    recursive_payload: dict,
    eval_id: str,
    evolution_payload: dict | None = None,
    judgment_sets: Iterable[JudgmentSet] | None = None,
    max_iterations: int = 1,
    command_limit: int | None = None,
    execute: bool = False,
    timeout_sec: int = 3600,
    apply_accepted: bool = False,
    writeback_manifests: bool = False,
) -> dict:
    """Run a bounded recursive executor loop."""

    store = JsonlGraphStore(graph_dir)
    current_recursive = recursive_payload
    normalized_judgments = list(judgment_sets or [])
    iterations = []
    applied_candidate_node_ids: list[str] = []
    daemon_manifests = []

    for index in range(max(1, max_iterations)):
        iteration_eval_id = f"{eval_id}_iter{index + 1}"
        execution_payload = build_recursive_execution_payload(
            root=root,
            graph_dir=graph_dir,
            recursive_payload=current_recursive,
            evolution_payload=evolution_payload,
            eval_id=iteration_eval_id,
            judgment_sets=normalized_judgments,
            command_limit=command_limit,
            execute=execute,
            timeout_sec=timeout_sec,
            include_full_resumed=True,
        )
        iteration_apply = _apply_if_requested(
            store=store,
            evolution_payload=evolution_payload,
            acceptance_payload=execution_payload.get("candidate_acceptance"),
            apply_accepted=apply_accepted,
        )
        applied_candidate_node_ids.extend(iteration_apply["applied_candidate_node_ids"])
        manifests = _iteration_manifests(
            eval_id=iteration_eval_id,
            execution_payload=execution_payload,
            apply_summary=iteration_apply,
        )
        daemon_manifests.extend(manifests)
        if writeback_manifests:
            for manifest in manifests:
                store.append_trial(manifest)
            store.flush()

        iterations.append({
            "iteration": index + 1,
            "eval_id": iteration_eval_id,
            "frontier": execution_payload.get("frontier", {}),
            "execution_status_counts": dict(Counter(r.get("status") for r in execution_payload.get("execution_records", []))),
            "candidate_acceptance_counts": (execution_payload.get("candidate_acceptance") or {}).get("decision_counts", {}),
            "apply_summary": iteration_apply,
            "resumed": bool(execution_payload.get("resumed_recursive_full")),
            "next_actions": (execution_payload.get("resumed_recursive") or {}).get("next_actions", []),
            "execution_payload": _compact_execution_payload(execution_payload),
        })

        resumed = execution_payload.get("resumed_recursive_full")
        if not resumed:
            break
        current_recursive = resumed
        normalized_judgments = []
        if not current_recursive.get("next_actions"):
            break

    return {
        "eval_id": eval_id,
        "mode": {
            "execute": execute,
            "timeout_sec": timeout_sec,
            "command_limit": command_limit,
            "max_iterations": max_iterations,
            "apply_accepted": apply_accepted,
            "writeback_manifests": writeback_manifests,
        },
        "source": {
            "recursive_eval_id": recursive_payload.get("eval_id"),
            "evolution_eval_id": (evolution_payload or {}).get("eval_id"),
            "graph_dir": str(graph_dir),
        },
        "iteration_count": len(iterations),
        "iterations": iterations,
        "applied_candidate_node_ids": sorted(set(applied_candidate_node_ids)),
        "manifest_count": len(daemon_manifests),
        "manifest_trial_ids": [m.trial_id for m in daemon_manifests],
        "manifests": [m.to_dict() for m in daemon_manifests],
    }


def build_preflight_queue_daemon_payload(
    *,
    root: Path,
    graph_dir: Path,
    preflight_payload: dict,
    eval_id: str,
    evolution_payload: dict | None = None,
    judgment_sets: Iterable[JudgmentSet] | None = None,
    resume_eval_id: str | None = None,
    queue_name: str = "preflight_queue",
    command_limit: int | None = None,
    execute: bool = False,
    timeout_sec: int = 3600,
    apply_accepted: bool = False,
    writeback_manifests: bool = False,
    artifact_baseline_variant: str | None = None,
) -> dict:
    """Convert a ready preflight queue into bounded daemon leaf work items."""

    store = JsonlGraphStore(graph_dir)
    normalized_judgment_sets = list(judgment_sets or [])
    artifact_eval = None
    artifact_judgment_sets: list[JudgmentSet] = []
    if artifact_baseline_variant:
        artifact_eval = build_queue_artifact_eval_payload(
            root=root,
            preflight_payload=preflight_payload,
            eval_id=f"{eval_id}_artifact_eval",
            baseline_variant=artifact_baseline_variant,
        )
        artifact_judgment_sets = judgment_sets_from_artifact_eval(artifact_eval)
        normalized_judgment_sets.extend(artifact_judgment_sets)
    queue_recursive = _recursive_payload_from_preflight_queue(
        preflight_payload=preflight_payload,
        eval_id=eval_id,
        queue_name=queue_name,
    )
    execution_payload = build_recursive_execution_payload(
        root=root,
        graph_dir=graph_dir,
        recursive_payload=queue_recursive,
        evolution_payload=evolution_payload,
        eval_id=f"{eval_id}_execute_queue",
        judgment_sets=normalized_judgment_sets,
        resume_eval_id=resume_eval_id or f"{eval_id}_resume",
        command_limit=command_limit,
        execute=execute,
        timeout_sec=timeout_sec,
        include_full_resumed=False,
    )
    apply_summary = _apply_if_requested(
        store=store,
        evolution_payload=evolution_payload,
        acceptance_payload=execution_payload.get("candidate_acceptance"),
        apply_accepted=apply_accepted,
    )
    manifests = _iteration_manifests(
        eval_id=eval_id,
        execution_payload=execution_payload,
        apply_summary=apply_summary,
    )
    if writeback_manifests:
        for manifest in manifests:
            store.append_trial(manifest)
        store.flush()
    status_counts = dict(Counter(record.get("status") for record in execution_payload.get("execution_records", [])))
    ready_rows = _ready_preflight_rows(preflight_payload)
    return {
        "eval_id": eval_id,
        "queue_name": queue_name,
        "mode": {
            "execute": execute,
            "timeout_sec": timeout_sec,
            "command_limit": command_limit,
            "writeback_manifests": writeback_manifests,
            "apply_accepted": apply_accepted,
            "judgment_sets": len(normalized_judgment_sets),
            "artifact_baseline_variant": artifact_baseline_variant,
            "artifact_auto_judgment_sets": len(artifact_judgment_sets),
        },
        "source": {
            "preflight_eval_id": preflight_payload.get("eval_id"),
            "graph_dir": str(graph_dir),
        },
        "ready_queue_count": len(ready_rows),
        "planned_leaf_count": execution_payload.get("frontier", {}).get("planned_actions", 0),
        "executable_leaf_count": execution_payload.get("frontier", {}).get("executable_actions", 0),
        "succeeded_leaf_count": status_counts.get("succeeded", 0),
        "failed_leaf_count": status_counts.get("failed", 0),
        "timeout_leaf_count": status_counts.get("timeout", 0),
        "execution_status_counts": status_counts,
        "proposal_ids": execution_payload.get("frontier", {}).get("proposal_ids", []),
        "candidate_acceptance_counts": (execution_payload.get("candidate_acceptance") or {}).get("decision_counts", {}),
        "accepted_proposal_ids": (execution_payload.get("candidate_acceptance") or {}).get("accepted_proposal_ids", []),
        "resumed": bool(execution_payload.get("resumed_recursive")),
        "resumed_next_action_count": len((execution_payload.get("resumed_recursive") or {}).get("next_actions", [])),
        "apply_summary": apply_summary,
        "applied_candidate_node_ids": apply_summary.get("applied_candidate_node_ids", []),
        "artifact_evaluation": artifact_eval,
        "execution_payload": _compact_execution_payload(execution_payload),
        "manifest_count": len(manifests),
        "manifest_trial_ids": [m.trial_id for m in manifests],
        "manifests": [m.to_dict() for m in manifests],
    }


def _recursive_payload_from_preflight_queue(*, preflight_payload: dict, eval_id: str, queue_name: str) -> dict:
    frames = []
    next_actions = []
    for row in _ready_preflight_rows(preflight_payload):
        proposal_id = str(row.get("proposal_id") or "")
        frame_id = stable_id("qframe", eval_id, proposal_id, row.get("command_hint", ""))
        problem_id = f"verify::{proposal_id}"
        frames.append({
            "frame_id": frame_id,
            "frame_type": "verification_subproblem",
            "status": "ready_to_act",
            "depth": 1,
            "problem_id": problem_id,
            "goal": "Run the queued fresh ablation command and return judgments to the parent proposal.",
            "hypothesis": "Preflight-ready proposals can be consumed as daemon leaf work items.",
            "expected_observation": "The command produces candidate outputs or judgments for later acceptance gating.",
            "verifier": "candidate_acceptance_gate",
            "parent_frame_id": stable_id("qframe", eval_id, queue_name, "root"),
            "assumption_ids": [proposal_id],
            "child_frame_ids": [],
            "source": {
                "proposal_id": proposal_id,
                "preflight_eval_id": preflight_payload.get("eval_id"),
                "queue_name": queue_name,
            },
            "argument": {
                "support": [
                    f"readiness={row.get('readiness')}",
                    f"trigger_rows={len(row.get('trigger_problem_ids', []))}",
                    f"control_rows={len(row.get('control_problem_ids', []))}",
                    "command_hint_present=True",
                ],
                "objections": [
                    "daemon queue execution is dry-run unless --execute is explicit",
                    "acceptance and graph mutation still require later judgment and apply gates",
                ],
                "falsification_tests": [
                    "command exists for every ready proposal",
                    "execution record is written before any graph mutation",
                ],
            },
            "next_action": "run_fresh_ablation",
            "command_hint": row.get("command_hint", ""),
            "return_update": {
                "on_success": "attach generated judgments to candidate acceptance and resume the parent frame",
                "on_failure": "record an execution_lapse residual and keep the parent unmutated",
            },
            "priority": float(row.get("priority", 0.0) or 0.0),
            "manifest": {},
        })
        next_actions.append({
            "frame_id": frame_id,
            "parent_frame_id": stable_id("qframe", eval_id, queue_name, "root"),
            "frame_type": "verification_subproblem",
            "status": "ready_to_act",
            "problem_id": problem_id,
            "proposal_id": proposal_id,
            "next_action": "run_fresh_ablation",
            "priority": float(row.get("priority", 0.0) or 0.0),
            "command_hint": row.get("command_hint", ""),
            "return_update": {
                "source_readiness": row.get("readiness"),
                "on_success": "candidate_acceptance_resume",
                "on_failure": "execution_lapse_residual",
            },
        })
    return {
        "eval_id": eval_id,
        "mode": {
            "source": "preflight_queue",
            "queue_name": queue_name,
            "source_preflight_eval_id": preflight_payload.get("eval_id"),
        },
        "root": {
            "problem_id": f"queue::{queue_name}",
            "problem": "Consume preflight-ready proposal commands as recursive daemon leaves.",
            "goal": "Create bounded executable leaf records without applying graph mutations.",
            "activated_assumption_ids": [],
        },
        "frame_counts": {"verification_subproblem": len(frames)},
        "status_counts": {"ready_to_act": len(frames)} if frames else {},
        "depth_counts": {"1": len(frames)} if frames else {},
        "recursion_edges": [],
        "open_frame_ids": [frame["frame_id"] for frame in frames],
        "next_actions": next_actions,
        "frames": frames,
    }


def _ready_preflight_rows(preflight_payload: dict) -> list[dict]:
    return [
        row
        for row in preflight_payload.get("summaries", [])
        if row.get("readiness") == "ready_for_fresh_ablation" and row.get("command_hint")
    ]


def _apply_if_requested(
    *,
    store: JsonlGraphStore,
    evolution_payload: dict | None,
    acceptance_payload: dict | None,
    apply_accepted: bool,
) -> dict:
    if not apply_accepted:
        return {
            "enabled": False,
            "reason": "apply_accepted disabled",
            "applied_candidate_node_ids": [],
        }
    if not evolution_payload or not acceptance_payload:
        return {
            "enabled": True,
            "reason": "missing evolution or acceptance payload",
            "applied_candidate_node_ids": [],
        }
    gated = _filter_acceptance_for_formal_gate(
        acceptance_payload,
        (evolution_payload or {}).get("formal_mapping_gate", {}),
    )
    applied = apply_accepted_candidates(
        store,
        evolution_payload.get("proposals", {}),
        gated,
    )
    return {
        "enabled": True,
        "accepted_proposal_ids": acceptance_payload.get("accepted_proposal_ids", []),
        "gated_accepted_proposal_ids": gated.get("accepted_proposal_ids", []),
        "applied_candidate_node_ids": applied,
        "reason": "applied accepted candidates" if applied else "no accepted candidates applied",
    }


def _filter_acceptance_for_formal_gate(acceptance_payload: dict, formal_gate_payload: dict) -> dict:
    blocked = {
        row.get("proposal_id")
        for row in formal_gate_payload.get("gates", [])
        if row.get("blocks_policy_update")
    }
    if not blocked:
        return acceptance_payload
    summaries = [
        row
        for row in acceptance_payload.get("summaries", [])
        if row.get("proposal_id") not in blocked
    ]
    return {
        **acceptance_payload,
        "accepted_proposal_ids": [
            pid for pid in acceptance_payload.get("accepted_proposal_ids", [])
            if pid not in blocked
        ],
        "summaries": summaries,
        "formal_gate_blocked_proposal_ids": sorted(blocked),
    }


def _iteration_manifests(*, eval_id: str, execution_payload: dict, apply_summary: dict) -> list:
    records = []
    for record in execution_payload.get("execution_records", []):
        status = record.get("status")
        manifest_status = TrialStatus.FAILED if status in {"failed", "timeout"} else TrialStatus.OBSERVED
        residual = None
        residual_type = None
        if status in {"failed", "timeout"}:
            residual = f"Recursive daemon command {status}: {record.get('reason', '')}"
            residual_type = ResidualType.EXECUTION_LAPSE
        records.append(make_component_manifest(
            eval_id=eval_id,
            event_type="tool_use",
            problem_id=f"daemon::{record.get('proposal_id') or record.get('frame_id')}",
            component="recursive_daemon",
            assumption="Frontier command execution/resume should advance a recursive assumption frame.",
            why_selected="The recursive runner exposed this frame as the current executable frontier.",
            expected_effect="Command or dry-run plan returns evidence needed by the parent frame.",
            assumption_ids=[record.get("proposal_id")] if record.get("proposal_id") else [],
            verifier="recursive_executor_record",
            verification_plan="Check return code, generated judgments, acceptance gate, and resumed next actions.",
            rollback_condition="Do not apply graph changes unless the acceptance gate passes.",
            status=manifest_status,
            observed_effect=f"execution_status={status}; reason={record.get('reason', '')}",
            residual=residual,
            residual_type=residual_type,
            artifacts={"execution_record": record},
            metadata={"execution_status": status},
            trial_id=stable_id("trial", eval_id, record.get("frame_id"), record.get("proposal_id"), status),
        ))
    records.append(make_component_manifest(
        eval_id=eval_id,
        event_type="recursive_daemon_iteration",
        problem_id=f"daemon_iteration::{eval_id}",
        component="recursive_daemon",
        assumption="A bounded execute-read-resume loop can advance recursive self-evolution without ungated graph mutation.",
        why_selected="The user requested automatic command execution, result reading, and looping.",
        expected_effect="Produce an iteration summary, optional acceptance resume, and explicit apply summary.",
        verifier="daemon_iteration_audit",
        verification_plan="Inspect planned actions, execution records, acceptance counts, and applied candidate ids.",
        rollback_condition="Disable --apply-accepted or revert candidate graph writes if acceptance evidence is wrong.",
        status=TrialStatus.OBSERVED,
        observed_effect=(
            f"planned={execution_payload.get('frontier', {}).get('planned_actions', 0)}; "
            f"resumed={bool(execution_payload.get('resumed_recursive_full'))}; "
            f"applied={len(apply_summary.get('applied_candidate_node_ids', []))}"
        ),
        artifacts={
            "frontier": execution_payload.get("frontier", {}),
            "acceptance_counts": (execution_payload.get("candidate_acceptance") or {}).get("decision_counts", {}),
            "apply_summary": apply_summary,
        },
        metadata={"apply_accepted": apply_summary.get("enabled", False)},
    ))
    return records


def _compact_execution_payload(payload: dict) -> dict:
    return {
        "eval_id": payload.get("eval_id"),
        "mode": payload.get("mode"),
        "frontier": payload.get("frontier"),
        "execution_records": payload.get("execution_records", []),
        "acceptance_error": payload.get("acceptance_error", ""),
        "candidate_acceptance": payload.get("candidate_acceptance"),
        "resumed_recursive": payload.get("resumed_recursive"),
    }


def _load_json(path: Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--recursive-payload", default=None)
    ap.add_argument("--preflight-payload", default=None)
    ap.add_argument("--evolution-payload", default=None)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--queue-name", default="preflight_queue")
    ap.add_argument("--judgment-bundle", default=None)
    ap.add_argument("--candidate-judgments", nargs="*", default=None)
    ap.add_argument("--candidate-variant", default=None)
    ap.add_argument("--candidate-baseline", default=None)
    ap.add_argument("--proposal-ids", nargs="*", default=None)
    ap.add_argument("--max-iterations", type=int, default=1)
    ap.add_argument("--command-limit", type=int, default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=3600)
    ap.add_argument("--apply-accepted", action="store_true")
    ap.add_argument("--writeback-manifests", action="store_true")
    ap.add_argument("--artifact-baseline-variant", default=None,
                    help="auto-discover cached fresh-ablation judgments against this baseline variant")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.preflight_payload:
        judgment_sets = load_judgment_sets(
            root=root,
            judgment_bundle=_resolve(root, args.judgment_bundle) if args.judgment_bundle else None,
            candidate_variant=args.candidate_variant,
            baseline_variant=args.candidate_baseline,
            judgment_paths=[_resolve(root, p) for p in args.candidate_judgments or []],
            proposal_ids=args.proposal_ids,
        )
        payload = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=_resolve(root, args.graph_dir),
            preflight_payload=_load_json(_resolve(root, args.preflight_payload)) or {},
            evolution_payload=_load_json(_resolve(root, args.evolution_payload)) if args.evolution_payload else None,
            judgment_sets=judgment_sets,
            eval_id=args.eval_id,
            queue_name=args.queue_name,
            command_limit=args.command_limit,
            execute=args.execute,
            timeout_sec=args.timeout_sec,
            apply_accepted=args.apply_accepted,
            writeback_manifests=args.writeback_manifests,
            artifact_baseline_variant=args.artifact_baseline_variant,
        )
    else:
        if not args.recursive_payload:
            raise SystemExit("--recursive-payload is required unless --preflight-payload is supplied")
        judgment_sets = load_judgment_sets(
            root=root,
            judgment_bundle=_resolve(root, args.judgment_bundle) if args.judgment_bundle else None,
            candidate_variant=args.candidate_variant,
            baseline_variant=args.candidate_baseline,
            judgment_paths=[_resolve(root, p) for p in args.candidate_judgments or []],
            proposal_ids=args.proposal_ids,
        )
        payload = build_recursive_daemon_payload(
            root=root,
            graph_dir=_resolve(root, args.graph_dir),
            recursive_payload=_load_json(_resolve(root, args.recursive_payload)) or {},
            evolution_payload=_load_json(_resolve(root, args.evolution_payload)),
            eval_id=args.eval_id,
            judgment_sets=judgment_sets,
            max_iterations=args.max_iterations,
            command_limit=args.command_limit,
            execute=args.execute,
            timeout_sec=args.timeout_sec,
            apply_accepted=args.apply_accepted,
            writeback_manifests=args.writeback_manifests,
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
