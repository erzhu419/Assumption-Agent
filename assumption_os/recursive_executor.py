"""Executor/resume layer for recursive assumption runs.

The recursive runner creates the task tree.  This module consumes the current
frontier, records executable leaf commands, optionally runs them, converts
fresh candidate judgments into an acceptance payload, and resumes the recursive
tree so child verification results update parent candidate frames.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .candidate_acceptance import build_acceptance_payload
from .recursive_runner import build_recursive_assumption_run


EXECUTABLE_NEXT_ACTIONS = {
    "run_fresh_ablation",
    "run_ablation",
    "run_fresh_ablation_before_promotion",
}


@dataclass(frozen=True)
class JudgmentSet:
    candidate_variant: str
    baseline_variant: str
    judgment_paths: list[Path]
    proposal_ids: list[str]


def build_recursive_execution_payload(
    *,
    root: Path,
    graph_dir: Path,
    recursive_payload: dict,
    eval_id: str,
    evolution_payload: dict | None = None,
    judgment_sets: Iterable[JudgmentSet] | None = None,
    resume_eval_id: str | None = None,
    command_limit: int | None = None,
    execute: bool = False,
    timeout_sec: int = 3600,
) -> dict:
    """Plan/execute recursive frontier actions and optionally resume the tree."""

    action_plan = _frontier_action_plan(recursive_payload, command_limit=command_limit)
    execution_records = _execute_action_plan(
        root=root,
        action_plan=action_plan,
        execute=execute,
        timeout_sec=timeout_sec,
    )
    acceptance_payload = None
    resumed_recursive_payload = None
    acceptance_error = ""
    normalized_judgment_sets = list(judgment_sets or [])

    if normalized_judgment_sets:
        if evolution_payload is None:
            acceptance_error = "judgment sets were supplied but no evolution payload was available"
        else:
            acceptance_payload = _build_combined_acceptance_payload(
                evolution_payload=evolution_payload,
                judgment_sets=normalized_judgment_sets,
                eval_id=f"{eval_id}_candidate_acceptance",
            )
            resumed_recursive_payload = _resume_recursive_payload(
                graph_dir=graph_dir,
                recursive_payload=recursive_payload,
                evolution_payload=evolution_payload,
                acceptance_payload=acceptance_payload,
                resume_eval_id=resume_eval_id or f"{eval_id}_resume",
            )

    return {
        "eval_id": eval_id,
        "mode": {
            "execute": execute,
            "timeout_sec": timeout_sec,
            "command_limit": command_limit,
            "judgment_sets": len(normalized_judgment_sets),
            "resumed": resumed_recursive_payload is not None,
        },
        "source": {
            "recursive_eval_id": recursive_payload.get("eval_id"),
            "evolution_eval_id": (evolution_payload or {}).get("eval_id"),
            "graph_dir": _display_path(root, graph_dir),
        },
        "frontier": {
            "next_actions": len(recursive_payload.get("next_actions", [])),
            "planned_actions": len(action_plan),
            "executable_actions": sum(1 for item in action_plan if item["executable"]),
            "action_counts": dict(Counter(item["next_action"] for item in action_plan)),
            "proposal_ids": [item["proposal_id"] for item in action_plan if item["proposal_id"]],
        },
        "action_plan": action_plan,
        "execution_records": execution_records,
        "acceptance_error": acceptance_error,
        "candidate_acceptance": acceptance_payload,
        "resumed_recursive": _compact_resumed_payload(resumed_recursive_payload),
    }


def _frontier_action_plan(recursive_payload: dict, *, command_limit: int | None) -> list[dict]:
    frames_by_id = {frame["frame_id"]: frame for frame in recursive_payload.get("frames", [])}
    actions = recursive_payload.get("next_actions", [])
    if command_limit is not None:
        actions = actions[:command_limit]
    planned = []
    for action in actions:
        frame = frames_by_id.get(action.get("frame_id", ""), {})
        proposal_id = _proposal_id(action, frame)
        command = action.get("command_hint", "")
        executable = bool(command) and action.get("next_action") in EXECUTABLE_NEXT_ACTIONS
        planned.append({
            "frame_id": action.get("frame_id"),
            "parent_frame_id": action.get("parent_frame_id"),
            "frame_type": action.get("frame_type"),
            "status": action.get("status"),
            "problem_id": action.get("problem_id"),
            "proposal_id": proposal_id,
            "next_action": action.get("next_action"),
            "priority": action.get("priority", 0.0),
            "executable": executable,
            "command": command,
            "reason": (
                "ready executable frontier command"
                if executable
                else "frontier action has no executable command_hint"
            ),
            "return_update": action.get("return_update", {}),
        })
    return planned


def _execute_action_plan(
    *,
    root: Path,
    action_plan: list[dict],
    execute: bool,
    timeout_sec: int,
) -> list[dict]:
    records = []
    for item in action_plan:
        if not item["executable"]:
            records.append(_execution_record(item, status="skipped", reason=item["reason"]))
            continue
        if not execute:
            records.append(_execution_record(item, status="planned", reason="dry-run execution disabled"))
            continue
        records.append(_run_command(root=root, item=item, timeout_sec=timeout_sec))
    return records


def _execution_record(item: dict, *, status: str, reason: str, **extra) -> dict:
    return {
        "frame_id": item.get("frame_id"),
        "proposal_id": item.get("proposal_id"),
        "next_action": item.get("next_action"),
        "status": status,
        "reason": reason,
        **extra,
    }


def _run_command(*, root: Path, item: dict, timeout_sec: int) -> dict:
    try:
        proc = subprocess.run(
            item["command"],
            cwd=root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _execution_record(
            item,
            status="timeout",
            reason=f"command exceeded timeout_sec={timeout_sec}",
            stdout=(exc.stdout or "")[-4000:],
            stderr=(exc.stderr or "")[-4000:],
        )
    return _execution_record(
        item,
        status="succeeded" if proc.returncode == 0 else "failed",
        reason=f"returncode={proc.returncode}",
        returncode=proc.returncode,
        stdout=proc.stdout[-4000:],
        stderr=proc.stderr[-4000:],
    )


def _build_combined_acceptance_payload(
    *,
    evolution_payload: dict,
    judgment_sets: list[JudgmentSet],
    eval_id: str,
) -> dict:
    summaries = []
    run_payloads = []
    for index, judgment_set in enumerate(judgment_sets):
        if not judgment_set.proposal_ids:
            raise ValueError("each judgment set must name at least one proposal_id")
        payload = build_acceptance_payload(
            proposal_payload=evolution_payload.get("proposals", {}),
            preflight_payload=evolution_payload.get("candidate_preflight", {}),
            judgment_paths=judgment_set.judgment_paths,
            candidate_variant=judgment_set.candidate_variant,
            baseline_variant=judgment_set.baseline_variant,
            eval_id=f"{eval_id}_set{index + 1}",
            proposal_ids=judgment_set.proposal_ids,
        )
        summaries.extend(payload.get("summaries", []))
        run_payloads.append({
            "eval_id": payload.get("eval_id"),
            "candidate_variant": payload.get("candidate_variant"),
            "baseline_variant": payload.get("baseline_variant"),
            "judgment_paths": [str(path) for path in judgment_set.judgment_paths],
            "proposal_ids": judgment_set.proposal_ids,
            "decision_counts": payload.get("decision_counts", {}),
            "accepted_proposal_ids": payload.get("accepted_proposal_ids", []),
        })
    decision_counts = Counter(row.get("decision") for row in summaries)
    accepted = [row["proposal_id"] for row in summaries if row.get("decision") == "accept"]
    return {
        "eval_id": eval_id,
        "runs": run_payloads,
        "decision_counts": dict(decision_counts),
        "accepted_proposal_ids": accepted,
        "summaries": summaries,
    }


def _resume_recursive_payload(
    *,
    graph_dir: Path,
    recursive_payload: dict,
    evolution_payload: dict,
    acceptance_payload: dict,
    resume_eval_id: str,
) -> dict:
    root_payload = recursive_payload.get("root", {})
    mode = recursive_payload.get("mode", {})
    return build_recursive_assumption_run(
        graph_dir=graph_dir,
        problem=root_payload.get("problem", ""),
        goal=root_payload.get("goal", ""),
        problem_id=root_payload.get("problem_id"),
        eval_id=resume_eval_id,
        evolution_payload=evolution_payload,
        acceptance_payload=acceptance_payload,
        top_k=int(mode.get("top_k", 6) or 6),
        max_children=int(mode.get("max_children", 8) or 8),
        max_depth=int(mode.get("max_depth", 3) or 3),
        writeback=False,
    )


def _compact_resumed_payload(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    return {
        "eval_id": payload.get("eval_id"),
        "mode": payload.get("mode"),
        "frame_counts": payload.get("frame_counts", {}),
        "status_counts": payload.get("status_counts", {}),
        "next_actions": payload.get("next_actions", []),
    }


def _proposal_id(action: dict, frame: dict) -> str:
    source = frame.get("source", {})
    if source.get("proposal_id"):
        return source["proposal_id"]
    problem_id = action.get("problem_id") or frame.get("problem_id", "")
    for prefix in ("verify::", "proposal::", "evidence::", "repair::"):
        if problem_id.startswith(prefix):
            return problem_id.removeprefix(prefix)
    return ""


def load_judgment_sets(
    *,
    root: Path,
    judgment_bundle: Path | None = None,
    candidate_variant: str | None = None,
    baseline_variant: str | None = None,
    judgment_paths: Iterable[Path] | None = None,
    proposal_ids: Iterable[str] | None = None,
) -> list[JudgmentSet]:
    """Load either a bundle of per-run judgments or one CLI judgment set."""

    if judgment_bundle:
        bundle = json.loads(judgment_bundle.read_text(encoding="utf-8"))
        out = []
        for run in bundle.get("runs", []):
            proposal_ids = list(run.get("proposal_ids", []))
            if not proposal_ids:
                raise ValueError("each --judgment-bundle run must include proposal_ids")
            out.append(JudgmentSet(
                candidate_variant=run["candidate_variant"],
                baseline_variant=run.get("baseline_variant") or bundle["baseline_variant"],
                judgment_paths=[_resolve(root, p) for p in run.get("judgments", [])],
                proposal_ids=proposal_ids,
            ))
        return out
    paths = list(judgment_paths or [])
    if not paths:
        return []
    if not candidate_variant or not baseline_variant:
        raise ValueError("--candidate-variant and --candidate-baseline are required when --candidate-judgments is supplied")
    if not proposal_ids:
        raise ValueError("--proposal-ids is required when --candidate-judgments is supplied")
    return [
        JudgmentSet(
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            judgment_paths=paths,
            proposal_ids=list(proposal_ids or []),
        )
    ]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--recursive-payload", required=True)
    ap.add_argument("--evolution-payload", default=None)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--resume-eval-id", default=None)
    ap.add_argument("--candidate-judgments", nargs="*", default=None)
    ap.add_argument("--candidate-variant", default=None)
    ap.add_argument("--candidate-baseline", default=None)
    ap.add_argument("--proposal-ids", nargs="*", default=None)
    ap.add_argument("--judgment-bundle", default=None)
    ap.add_argument("--command-limit", type=int, default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=3600)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    recursive_payload = _load_json(_resolve(root, args.recursive_payload))
    evolution_payload = (
        _load_json(_resolve(root, args.evolution_payload))
        if args.evolution_payload
        else None
    )
    judgment_sets = load_judgment_sets(
        root=root,
        judgment_bundle=_resolve(root, args.judgment_bundle) if args.judgment_bundle else None,
        candidate_variant=args.candidate_variant,
        baseline_variant=args.candidate_baseline,
        judgment_paths=[_resolve(root, p) for p in args.candidate_judgments or []],
        proposal_ids=args.proposal_ids,
    )
    payload = build_recursive_execution_payload(
        root=root,
        graph_dir=_resolve(root, args.graph_dir),
        recursive_payload=recursive_payload,
        evolution_payload=evolution_payload,
        eval_id=args.eval_id,
        resume_eval_id=args.resume_eval_id,
        judgment_sets=judgment_sets,
        command_limit=args.command_limit,
        execute=args.execute,
        timeout_sec=args.timeout_sec,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
