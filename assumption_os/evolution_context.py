"""Evolution-context and harness-responsibility gate.

This module treats the self-evolution procedure itself as an auditable
assumption.  A candidate graph edit is not only judged by its local acceptance
payload; the surrounding harness must also declare the task, permissions,
observability, verification, failure attribution, rollback path, and
intervention record.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore
from .schema import AssumptionType, TrialManifest, TrialStatus, stable_id


class ResponsibilityStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class EvolutionPolicyDecision(str, Enum):
    DRY_RUN_READY = "dry_run_ready"
    READY_FOR_MANUAL_APPLY = "ready_for_manual_apply"
    GATED_APPLY_ALLOWED = "gated_apply_allowed"
    WRITEBACK_ONLY = "writeback_only"
    BLOCKED_BY_PERMISSIONS = "blocked_by_permissions"
    BLOCKED_BY_HARNESS = "blocked_by_harness"


@dataclass(frozen=True)
class HarnessResponsibilityCheck:
    name: str
    status: ResponsibilityStatus
    required: bool
    evidence: dict = field(default_factory=dict)
    rationale: str = ""
    remediation: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class EvolutionPermission:
    allow_writeback: bool = False
    allow_apply_accepted: bool = False
    allow_execute_commands: bool = False
    allow_autonomous_apply: bool = False
    max_apply_candidates: int = 0
    max_frontier_commands: int = 0
    allowed_surfaces: list[str] = field(default_factory=lambda: [
        "assumption_graph",
        "retrieval_policy",
        "verifier_stack",
        "world_model",
        "recursive_runner",
        "harness",
    ])
    forbidden_surfaces: list[str] = field(default_factory=list)
    human_review_required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_RESPONSIBILITIES = [
    "task_specification",
    "context_selection",
    "memory_observability",
    "failure_attribution",
    "verification_protocol",
    "permission_boundary",
    "intervention_recording",
    "rollback_contract",
    "evolution_procedure",
]


def build_evolution_context_payload(
    *,
    eval_id: str,
    objective: str,
    sections: dict[str, dict] | None = None,
    mode: dict | None = None,
    permissions: dict | EvolutionPermission | None = None,
    source_payloads: dict[str, dict] | None = None,
) -> dict:
    """Build a harness-responsibility payload for one evolution step."""

    sections = sections or {}
    mode = _default_mode(mode)
    permission = permissions if isinstance(permissions, EvolutionPermission) else EvolutionPermission(**(permissions or {}))
    source_payloads = source_payloads or {}
    checks = _responsibility_checks(
        objective=objective,
        sections=sections,
        mode=mode,
        permissions=permission,
        source_payloads=source_payloads,
    )
    violations = _permission_violations(
        mode=mode,
        permissions=permission,
        sections=sections,
    )
    decision = _policy_decision(
        checks=checks,
        violations=violations,
        mode=mode,
        permissions=permission,
        sections=sections,
    )
    procedure_updates = _procedure_updates(
        checks=checks,
        decision=decision,
        mode=mode,
        permissions=permission,
        sections=sections,
    )
    manifest = _context_manifest(
        eval_id=eval_id,
        objective=objective,
        decision=decision,
        checks=checks,
        violations=violations,
        procedure_updates=procedure_updates,
    )
    return {
        "eval_id": eval_id,
        "objective": objective,
        "mode": mode,
        "permissions": permission.to_dict(),
        "source_eval_ids": _source_eval_ids(sections=sections, source_payloads=source_payloads),
        "responsibility_count": len(checks),
        "responsibility_status_counts": dict(Counter(check.status.value for check in checks)),
        "responsibilities": [check.to_dict() for check in checks],
        "permission_violations": violations,
        "accepted_candidate_count": _accepted_candidate_count(sections),
        "actionable_frontier_count": _actionable_frontier_count(sections),
        "policy_decision": decision.value,
        "procedure_updates": procedure_updates,
        "manifest": manifest.to_dict(),
    }


def write_evolution_context_manifest(
    *,
    graph_dir: Path,
    context_payload: dict,
) -> str:
    manifest = TrialManifest.from_dict(context_payload["manifest"])
    store = JsonlGraphStore(graph_dir)
    store.append_trial(manifest)
    store.flush()
    return manifest.trial_id


def _responsibility_checks(
    *,
    objective: str,
    sections: dict[str, dict],
    mode: dict,
    permissions: EvolutionPermission,
    source_payloads: dict[str, dict],
) -> list[HarnessResponsibilityCheck]:
    checks = [
        _check(
            name="task_specification",
            ok=bool(objective.strip()),
            evidence={"objective_present": bool(objective.strip()), "objective": objective},
            rationale="The evolution run has an explicit objective.",
            remediation="Provide a concrete objective before generating or applying evolution actions.",
        ),
        _check(
            name="context_selection",
            ok=(
                _section_pass(sections, "trajectory_search")
                and _proposal_count(sections) > 0
            ),
            evidence={
                "trajectory_search_pass": _section_pass(sections, "trajectory_search"),
                "proposal_count": _proposal_count(sections),
                "multi_path_rate": sections.get("trajectory_search", {}).get("multi_path_rate"),
            },
            rationale="Candidate futures are selected with multi-path context instead of a single deterministic route.",
            remediation="Run trajectory_search and verifier_stack before allowing evolution actions.",
        ),
        _check(
            name="memory_observability",
            ok=(
                _section_pass(sections, "manifest_logger")
                and _section_pass(sections, "harness_observer")
                and not sections.get("manifest_logger", {}).get("secret_leak_detected", True)
                and sections.get("harness_observer", {}).get("full_coverage_after_writeback", False)
            ),
            evidence={
                "manifest_logger_pass": _section_pass(sections, "manifest_logger"),
                "harness_observer_pass": _section_pass(sections, "harness_observer"),
                "event_count": sections.get("manifest_logger", {}).get("event_count", 0),
                "artifact_coverage": sections.get("harness_observer", {}).get("full_coverage_after_writeback", False),
                "secret_leak_detected": sections.get("manifest_logger", {}).get("secret_leak_detected"),
            },
            rationale="LLM/retrieval/judge/tool events and harness artifacts are observable and redacted.",
            remediation="Backfill or log component manifests and fix redaction before mutation.",
        ),
        _check(
            name="failure_attribution",
            ok=_section_pass(sections, "residual_clusterer"),
            evidence={
                "residual_clusterer_pass": _section_pass(sections, "residual_clusterer"),
                "cluster_count": sections.get("residual_clusterer", {}).get("cluster_count", 0),
                "proposal_count": sections.get("residual_clusterer", {}).get("proposal_count", 0),
            },
            rationale="Systematic residuals are clustered before new method hypotheses are promoted.",
            remediation="Run residual clustering or record why failure attribution is not needed.",
        ),
        _check(
            name="verification_protocol",
            ok=(
                _section_pass(sections, "verifier_stack")
                and _section_pass(sections, "world_model")
                and _section_pass(sections, "formal_metrics")
                and sections.get("verifier_stack", {}).get("accepted_protocol_ok", False)
                and sections.get("verifier_stack", {}).get("rejected_protocol_ok", False)
            ),
            evidence={
                "verifier_stack_pass": _section_pass(sections, "verifier_stack"),
                "world_model_pass": _section_pass(sections, "world_model"),
                "formal_metrics_pass": _section_pass(sections, "formal_metrics"),
                "accepted_protocol_ok": sections.get("verifier_stack", {}).get("accepted_protocol_ok"),
                "rejected_protocol_ok": sections.get("verifier_stack", {}).get("rejected_protocol_ok"),
                "falsification_experiment_count": sections.get("verifier_stack", {}).get("falsification_experiment_count", 0),
            },
            rationale="Promotion decisions have world-model, formal, and falsification evidence.",
            remediation="Run verifier_stack with falsification protocols and formal/world-model checks.",
        ),
        _check(
            name="permission_boundary",
            ok=not _permission_violations(mode=mode, permissions=permissions, sections=sections),
            evidence={
                "mode": mode,
                "permissions": permissions.to_dict(),
                "accepted_candidate_count": _accepted_candidate_count(sections),
            },
            rationale="Requested operations stay inside declared write/apply/execute permissions.",
            remediation="Either lower requested mutation mode or explicitly authorize a bounded gated apply.",
        ),
        _check(
            name="intervention_recording",
            ok=(
                _section_pass(sections, "recursive_audit")
                and _section_pass(sections, "recursive_daemon")
                and sections.get("recursive_daemon", {}).get("accepted_apply_count", 0) >= 0
            ),
            evidence={
                "recursive_audit_pass": _section_pass(sections, "recursive_audit"),
                "recursive_daemon_pass": _section_pass(sections, "recursive_daemon"),
                "closure_score": sections.get("recursive_audit", {}).get("min_closure_score"),
                "daemon_cases": sections.get("recursive_daemon", {}).get("case_count"),
            },
            rationale="Recursive execution has an auditable frontier and daemon intervention records.",
            remediation="Run recursive_audit and daemon dry-run before applying accepted candidates.",
        ),
        _check(
            name="rollback_contract",
            ok=(
                _section_pass(sections, "recursive_audit")
                and _section_pass(sections, "verifier_stack")
                and sections.get("recursive_audit", {}).get("critical_issue_count", 1) == 0
                and _accepted_candidate_count(sections) <= max(permissions.max_apply_candidates, _accepted_candidate_count(sections))
            ),
            evidence={
                "critical_recursive_issues": sections.get("recursive_audit", {}).get("critical_issue_count"),
                "warning_recursive_issues": sections.get("recursive_audit", {}).get("warning_issue_count"),
                "accepted_candidate_count": _accepted_candidate_count(sections),
                "max_apply_candidates": permissions.max_apply_candidates,
            },
            rationale="Accepted candidates have parent-return and rollback-sensitive verifier records.",
            remediation="Add return_update contracts and cap apply count before graph mutation.",
        ),
        _check(
            name="evolution_procedure",
            ok=bool(source_payloads or sections),
            evidence={
                "section_count": len(sections),
                "source_payload_count": len(source_payloads),
                "mode": mode,
            },
            rationale="The evolution procedure is itself represented as input context, not implicit CLI state only.",
            remediation="Pass the current performance/evolution/recursive/verifier payload into the context gate.",
        ),
    ]
    return checks


def _check(
    *,
    name: str,
    ok: bool,
    evidence: dict,
    rationale: str,
    remediation: str,
    required: bool = True,
) -> HarnessResponsibilityCheck:
    return HarnessResponsibilityCheck(
        name=name,
        status=ResponsibilityStatus.PASS if ok else ResponsibilityStatus.FAIL,
        required=required,
        evidence=evidence,
        rationale=rationale if ok else "Responsibility is not satisfied.",
        remediation="" if ok else remediation,
    )


def _permission_violations(
    *,
    mode: dict,
    permissions: EvolutionPermission,
    sections: dict[str, dict],
) -> list[dict]:
    violations = []
    if mode.get("writeback") and not permissions.allow_writeback:
        violations.append({"kind": "writeback_not_allowed", "requested": True})
    if mode.get("execute_commands") and not permissions.allow_execute_commands:
        violations.append({"kind": "execute_commands_not_allowed", "requested": True})
    if mode.get("autonomous_apply") and not permissions.allow_autonomous_apply:
        violations.append({"kind": "autonomous_apply_not_allowed", "requested": True})
    if mode.get("apply_accepted") and not permissions.allow_apply_accepted:
        violations.append({"kind": "apply_accepted_not_allowed", "requested": True})
    accepted_count = _accepted_candidate_count(sections)
    if mode.get("apply_accepted") and accepted_count > permissions.max_apply_candidates:
        violations.append({
            "kind": "apply_candidate_budget_exceeded",
            "accepted_candidate_count": accepted_count,
            "max_apply_candidates": permissions.max_apply_candidates,
        })
    if mode.get("execute_commands"):
        frontier = _actionable_frontier_count(sections)
        if frontier > permissions.max_frontier_commands:
            violations.append({
                "kind": "frontier_command_budget_exceeded",
                "actionable_frontier_count": frontier,
                "max_frontier_commands": permissions.max_frontier_commands,
            })
    return violations


def _policy_decision(
    *,
    checks: list[HarnessResponsibilityCheck],
    violations: list[dict],
    mode: dict,
    permissions: EvolutionPermission,
    sections: dict[str, dict],
) -> EvolutionPolicyDecision:
    required_failures = [check for check in checks if check.required and check.status == ResponsibilityStatus.FAIL]
    if violations:
        return EvolutionPolicyDecision.BLOCKED_BY_PERMISSIONS
    if required_failures:
        return EvolutionPolicyDecision.BLOCKED_BY_HARNESS
    if mode.get("apply_accepted"):
        return EvolutionPolicyDecision.GATED_APPLY_ALLOWED
    if mode.get("writeback"):
        return EvolutionPolicyDecision.WRITEBACK_ONLY
    if _accepted_candidate_count(sections):
        return EvolutionPolicyDecision.READY_FOR_MANUAL_APPLY
    return EvolutionPolicyDecision.DRY_RUN_READY


def _procedure_updates(
    *,
    checks: list[HarnessResponsibilityCheck],
    decision: EvolutionPolicyDecision,
    mode: dict,
    permissions: EvolutionPermission,
    sections: dict[str, dict],
) -> list[dict]:
    updates = []
    failed = {check.name for check in checks if check.status == ResponsibilityStatus.FAIL}
    if "verification_protocol" not in failed:
        updates.append({
            "id": "require_verifier_stack_before_apply",
            "surface": "harness",
            "status": "active",
            "rule": "Accepted candidates can be applied only after verifier_stack passes with accepted/rejected protocol checks.",
        })
    if "intervention_recording" not in failed:
        updates.append({
            "id": "require_recursive_audit_before_daemon_apply",
            "surface": "recursive_runner",
            "status": "active",
            "rule": "Recursive daemon apply requires recursive_audit closure score >= 0.9 and no critical issues.",
        })
    if "memory_observability" not in failed:
        updates.append({
            "id": "require_manifest_and_harness_coverage",
            "surface": "observability",
            "status": "active",
            "rule": "LLM/retrieval/judge/tool events and harness artifacts must have redacted manifest coverage.",
        })
    if decision == EvolutionPolicyDecision.BLOCKED_BY_PERMISSIONS:
        updates.append({
            "id": "lower_mutation_mode_or_expand_permission",
            "surface": "permissions",
            "status": "blocked",
            "rule": "Requested mutation exceeds declared permission boundary.",
        })
    if _accepted_candidate_count(sections) and not mode.get("apply_accepted"):
        updates.append({
            "id": "manual_apply_available",
            "surface": "assumption_graph",
            "status": "ready",
            "rule": "Accepted candidates exist but require an explicit apply_accepted request.",
        })
    return updates


def _context_manifest(
    *,
    eval_id: str,
    objective: str,
    decision: EvolutionPolicyDecision,
    checks: list[HarnessResponsibilityCheck],
    violations: list[dict],
    procedure_updates: list[dict],
) -> TrialManifest:
    failed = [check.name for check in checks if check.status == ResponsibilityStatus.FAIL]
    manifest = TrialManifest(
        problem_id=f"evolution_context::{eval_id}",
        action_type="evolution_context_gate",
        component="evolution_context",
        assumption="The self-evolution step should be governed by explicit harness responsibilities and permission boundaries.",
        why_selected="Evolution procedure/context is itself an assumption that can create regressions if left implicit.",
        expected_effect="Return a bounded policy decision and procedure-update plan before graph mutation.",
        assumption_ids=[],
        predicted_regressions=[
            "silent graph mutation",
            "unlogged harness edits",
            "accepted candidate applied without rollback contract",
        ],
        verifier="evolution_context_responsibility_gate",
        verification_plan="All required harness responsibilities pass and requested mutations stay within permissions.",
        rollback_condition="Block apply/write/execute if required responsibilities fail or permissions are exceeded.",
        status=TrialStatus.OBSERVED if not failed and not violations else TrialStatus.FAILED,
        observed_effect=f"policy_decision={decision.value}; failed={failed}; violations={len(violations)}",
        residual="; ".join(failed) if failed else None,
        artifacts={
            "decision": decision.value,
            "failed_responsibilities": failed,
            "permission_violations": violations,
            "procedure_updates": procedure_updates,
        },
        metadata={
            "eval_id": eval_id,
            "node_type": AssumptionType.HARNESS.value,
            "responsibility_count": len(checks),
            "objective": objective,
        },
        trial_id=stable_id("trial", eval_id, "evolution_context"),
    )
    return manifest


def _default_mode(mode: dict | None) -> dict:
    out = {
        "writeback": False,
        "apply_accepted": False,
        "execute_commands": False,
        "autonomous_apply": False,
    }
    out.update(mode or {})
    return out


def _section_pass(sections: dict[str, dict], name: str) -> bool:
    return bool(sections.get(name, {}).get("pass"))


def _proposal_count(sections: dict[str, dict]) -> int:
    return int(sections.get("verifier_stack", {}).get("proposal_count") or 0)


def _accepted_candidate_count(sections: dict[str, dict]) -> int:
    return int(sections.get("verifier_stack", {}).get("accepted_count") or 0)


def _actionable_frontier_count(sections: dict[str, dict]) -> int:
    return int(sections.get("recursive_audit", {}).get("actionable_count") or 0)


def _source_eval_ids(*, sections: dict[str, dict], source_payloads: dict[str, dict]) -> dict:
    out = {}
    for name, payload in source_payloads.items():
        if isinstance(payload, dict) and payload.get("eval_id"):
            out[name] = payload["eval_id"]
    for name, section in sections.items():
        if isinstance(section, dict) and section.get("eval_id"):
            out[name] = section["eval_id"]
    return out


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
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--performance-payload", default=None)
    ap.add_argument("--sections-payload", default=None)
    ap.add_argument("--allow-writeback", action="store_true")
    ap.add_argument("--allow-apply-accepted", action="store_true")
    ap.add_argument("--allow-execute-commands", action="store_true")
    ap.add_argument("--allow-autonomous-apply", action="store_true")
    ap.add_argument("--max-apply-candidates", type=int, default=0)
    ap.add_argument("--max-frontier-commands", type=int, default=0)
    ap.add_argument("--writeback", action="store_true")
    ap.add_argument("--apply-accepted", action="store_true")
    ap.add_argument("--execute-commands", action="store_true")
    ap.add_argument("--autonomous-apply", action="store_true")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--writeback-manifest", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    perf = _load_json(_resolve(root, args.performance_payload))
    sections_payload = _load_json(_resolve(root, args.sections_payload))
    sections = {}
    if perf:
        sections = perf.get("sections", perf)
    if sections_payload:
        sections.update(sections_payload.get("sections", sections_payload))
    payload = build_evolution_context_payload(
        eval_id=args.eval_id,
        objective=args.objective,
        sections=sections,
        mode={
            "writeback": args.writeback,
            "apply_accepted": args.apply_accepted,
            "execute_commands": args.execute_commands,
            "autonomous_apply": args.autonomous_apply,
        },
        permissions=EvolutionPermission(
            allow_writeback=args.allow_writeback,
            allow_apply_accepted=args.allow_apply_accepted,
            allow_execute_commands=args.allow_execute_commands,
            allow_autonomous_apply=args.allow_autonomous_apply,
            max_apply_candidates=args.max_apply_candidates,
            max_frontier_commands=args.max_frontier_commands,
        ),
        source_payloads={"performance": perf or {}, "sections": sections_payload or {}},
    )
    if args.writeback_manifest:
        payload["written_manifest_trial_id"] = write_evolution_context_manifest(
            graph_dir=_resolve(root, args.graph_dir) or root,
            context_payload=payload,
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
