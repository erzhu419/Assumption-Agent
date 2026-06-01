"""Audit recursive assumption runner payloads for closed-loop structure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecursiveAuditIssue:
    severity: str
    check: str
    frame_id: str
    detail: str
    remediation: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_recursive_audit_payload(*, recursive_payload: dict, eval_id: str) -> dict:
    """Return structural and behavioral closure metrics for a recursive run."""

    frames = recursive_payload.get("frames", [])
    frames_by_id = {frame.get("frame_id"): frame for frame in frames if frame.get("frame_id")}
    issues: list[RecursiveAuditIssue] = []
    _audit_unique_frame_ids(frames, frames_by_id, issues)
    _audit_parent_child_consistency(recursive_payload, frames, frames_by_id, issues)
    _audit_frame_contracts(frames, frames_by_id, issues)
    _audit_frontier(recursive_payload, frames_by_id, issues)

    frame_count = len(frames)
    recursive_frame_count = sum(1 for frame in frames if frame.get("frame_type") != "root_problem")
    argument_complete_count = sum(1 for frame in frames if _has_argument_contract(frame))
    return_contract_count = sum(1 for frame in frames if _has_return_contract(frame))
    actionable_count = len(recursive_payload.get("next_actions", []))
    issue_counts = Counter(issue.severity for issue in issues)
    critical_count = issue_counts.get("critical", 0)
    warning_count = issue_counts.get("warning", 0)
    checks_total = max(1, frame_count * 5)
    checks_failed = critical_count + warning_count
    closure_score = max(0.0, 1.0 - checks_failed / checks_total)
    pass_ok = (
        frame_count > 0
        and critical_count == 0
        and closure_score >= 0.9
        and (argument_complete_count / frame_count) >= 0.85
        and (recursive_frame_count == 0 or return_contract_count / recursive_frame_count >= 0.75)
        and actionable_count >= 1
    )

    return {
        "eval_id": eval_id,
        "source_recursive_eval_id": recursive_payload.get("eval_id"),
        "pass": pass_ok,
        "frame_count": frame_count,
        "recursive_frame_count": recursive_frame_count,
        "argument_complete_count": argument_complete_count,
        "return_contract_count": return_contract_count,
        "actionable_count": actionable_count,
        "closure_score": round(closure_score, 4),
        "issue_counts": dict(issue_counts),
        "critical_issue_count": critical_count,
        "warning_issue_count": warning_count,
        "frame_type_counts": dict(Counter(frame.get("frame_type") for frame in frames)),
        "status_counts": dict(Counter(frame.get("status") for frame in frames)),
        "declared_edge_count": len(recursive_payload.get("recursion_edges", [])),
        "reconstructed_edge_count": len(_reconstruct_edges(frames, frames_by_id)),
        "next_gap_summary": _next_gap_summary(issues, recursive_payload),
        "issues": [issue.to_dict() for issue in issues],
    }


def _audit_unique_frame_ids(frames: list[dict], frames_by_id: dict[str, dict], issues: list[RecursiveAuditIssue]) -> None:
    frame_ids = [frame.get("frame_id") for frame in frames if frame.get("frame_id")]
    duplicates = [frame_id for frame_id, count in Counter(frame_ids).items() if count > 1]
    for frame_id in duplicates:
        issues.append(RecursiveAuditIssue(
            severity="critical",
            check="unique_frame_id",
            frame_id=frame_id,
            detail="Multiple recursive frames share the same frame_id.",
            remediation="Regenerate stable ids with a distinguishing frame kind or proposal id.",
        ))
    if len(frames_by_id) != len(frame_ids):
        issues.append(RecursiveAuditIssue(
            severity="critical",
            check="frame_index",
            frame_id="",
            detail="At least one recursive frame is missing frame_id.",
            remediation="Require frame_id before a frame can enter the recursive payload.",
        ))


def _audit_parent_child_consistency(
    recursive_payload: dict,
    frames: list[dict],
    frames_by_id: dict[str, dict],
    issues: list[RecursiveAuditIssue],
) -> None:
    declared_edges = {
        (edge.get("parent_frame_id"), edge.get("child_frame_id"))
        for edge in recursive_payload.get("recursion_edges", [])
    }
    reconstructed_edges = _reconstruct_edges(frames, frames_by_id)
    for frame in frames:
        frame_id = frame.get("frame_id", "")
        parent_id = frame.get("parent_frame_id")
        if frame.get("frame_type") != "root_problem" and not parent_id:
            issues.append(RecursiveAuditIssue(
                severity="critical",
                check="parent_link",
                frame_id=frame_id,
                detail="Non-root frame has no parent_frame_id.",
                remediation="Attach every recursive subproblem to the frame it updates.",
            ))
        if parent_id and parent_id not in frames_by_id:
            issues.append(RecursiveAuditIssue(
                severity="critical",
                check="parent_exists",
                frame_id=frame_id,
                detail=f"Parent frame {parent_id} is missing from the payload.",
                remediation="Persist parent frames and child frames in the same recursive payload.",
            ))
    missing_declared = reconstructed_edges - declared_edges
    stale_declared = declared_edges - reconstructed_edges
    for parent_id, child_id in sorted(missing_declared):
        issues.append(RecursiveAuditIssue(
            severity="warning",
            check="declared_recursion_edge",
            frame_id=child_id or "",
            detail=f"Edge {parent_id}->{child_id} is implied by frame links but missing from recursion_edges.",
            remediation="Regenerate recursion_edges from child_frame_ids before writing the payload.",
        ))
    for parent_id, child_id in sorted(stale_declared):
        issues.append(RecursiveAuditIssue(
            severity="warning",
            check="stale_recursion_edge",
            frame_id=child_id or "",
            detail=f"Edge {parent_id}->{child_id} is declared but not supported by frame links.",
            remediation="Drop stale recursion_edges or repair child_frame_ids/parent_frame_id.",
        ))


def _audit_frame_contracts(frames: list[dict], frames_by_id: dict[str, dict], issues: list[RecursiveAuditIssue]) -> None:
    for frame in frames:
        frame_id = frame.get("frame_id", "")
        for field in ["hypothesis", "expected_observation", "verifier", "next_action"]:
            if not frame.get(field):
                issues.append(RecursiveAuditIssue(
                    severity="critical" if field in {"hypothesis", "verifier"} else "warning",
                    check=f"frame_{field}",
                    frame_id=frame_id,
                    detail=f"Frame is missing {field}.",
                    remediation="Every recursive frame must state what it assumes and how it can be updated.",
                ))
        if not _has_argument_contract(frame):
            issues.append(RecursiveAuditIssue(
                severity="warning",
                check="argument_contract",
                frame_id=frame_id,
                detail="Frame argument is missing support, objections, or falsification_tests.",
                remediation="Record at least one supporting reason, objection, and falsifier per recursive frame.",
            ))
        if frame.get("frame_type") != "root_problem" and not _has_return_contract(frame):
            issues.append(RecursiveAuditIssue(
                severity="warning",
                check="return_contract",
                frame_id=frame_id,
                detail="Frame has no explicit return_update contract for its parent.",
                remediation="Add on_success/on_failure or parent_next_action so evidence can update the parent frame.",
            ))
        for child_id in frame.get("child_frame_ids", []):
            child = frames_by_id.get(child_id)
            if child and child.get("parent_frame_id") != frame_id:
                issues.append(RecursiveAuditIssue(
                    severity="critical",
                    check="bidirectional_child_link",
                    frame_id=child_id,
                    detail=f"Child does not point back to parent frame {frame_id}.",
                    remediation="Keep child_frame_ids and parent_frame_id bidirectionally consistent.",
                ))


def _audit_frontier(recursive_payload: dict, frames_by_id: dict[str, dict], issues: list[RecursiveAuditIssue]) -> None:
    next_actions = recursive_payload.get("next_actions", [])
    if not next_actions:
        issues.append(RecursiveAuditIssue(
            severity="warning",
            check="frontier",
            frame_id="",
            detail="Recursive payload has no actionable frontier.",
            remediation="Expose at least one open/ready frame as next_actions unless the run is fully resolved.",
        ))
    for action in next_actions:
        frame_id = action.get("frame_id", "")
        frame = frames_by_id.get(frame_id)
        if not frame:
            issues.append(RecursiveAuditIssue(
                severity="critical",
                check="frontier_frame_exists",
                frame_id=frame_id,
                detail="Frontier action points to a missing frame.",
                remediation="Build next_actions from the final frame table.",
            ))
            continue
        if frame.get("next_action") != action.get("next_action"):
            issues.append(RecursiveAuditIssue(
                severity="warning",
                check="frontier_next_action_sync",
                frame_id=frame_id,
                detail="Frontier action does not match the source frame next_action.",
                remediation="Regenerate next_actions after mutating or resuming frames.",
            ))


def _reconstruct_edges(frames: list[dict], frames_by_id: dict[str, dict]) -> set[tuple[str | None, str | None]]:
    edges = set()
    for frame in frames:
        frame_id = frame.get("frame_id")
        parent_id = frame.get("parent_frame_id")
        if parent_id:
            edges.add((parent_id, frame_id))
        for child_id in frame.get("child_frame_ids", []):
            if child_id in frames_by_id:
                edges.add((frame_id, child_id))
    return edges


def _has_argument_contract(frame: dict) -> bool:
    argument = frame.get("argument") or {}
    return bool(argument.get("support")) and "objections" in argument and bool(argument.get("falsification_tests"))


def _has_return_contract(frame: dict) -> bool:
    if frame.get("return_update", {}).get("parent_next_action"):
        return True
    return_update = frame.get("return_update") or {}
    return bool(return_update.get("on_success") and return_update.get("on_failure"))


def _next_gap_summary(issues: list[RecursiveAuditIssue], recursive_payload: dict) -> dict:
    if issues:
        top = Counter(issue.check for issue in issues).most_common(3)
        return {
            "status": "repair_required",
            "top_gaps": [{"check": check, "count": count} for check, count in top],
        }
    return {
        "status": "closed_loop_ready",
        "next_actions": [
            {
                "frame_id": action.get("frame_id"),
                "next_action": action.get("next_action"),
                "command_hint_present": bool(action.get("command_hint")),
            }
            for action in recursive_payload.get("next_actions", [])[:5]
        ],
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--recursive-payload", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_recursive_audit_payload(
        recursive_payload=_load_json(_resolve(root, args.recursive_payload)),
        eval_id=args.eval_id,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
