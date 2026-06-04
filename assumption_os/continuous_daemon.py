"""Budgeted continuous autonomy audit for the recursive daemon.

The recursive daemon intentionally stays gated: it may execute/read/resume
frontier work, but graph mutation still requires accepted candidates and an
explicit apply gate.  This module audits whether those bounded pieces compose
into a continuous loop without claiming an unbounded background service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_continuous_daemon_autonomy_payload(
    *,
    eval_id: str,
    recursive_daemon_section: dict,
    max_cycles: int = 5,
) -> dict:
    """Build a paper-facing audit for budgeted daemon autonomy."""

    cycles = _cycles(recursive_daemon_section)
    selected_cycles = cycles[: max(0, max_cycles)]
    gate_map = {
        "frontier_queue_coverage": (
            bool(recursive_daemon_section.get("preflight_queue_consumed"))
            and int(recursive_daemon_section.get("preflight_queue_ready_count") or 0) >= 5
            and int(recursive_daemon_section.get("preflight_queue_planned_leaf_count") or 0)
            == int(recursive_daemon_section.get("preflight_queue_ready_count") or 0)
        ),
        "bounded_execute_read_resume": (
            int(recursive_daemon_section.get("bounded_execute_succeeded_leaf_count") or 0) >= 1
            and int(recursive_daemon_section.get("bounded_execute_accept_count") or 0) >= 1
            and bool(recursive_daemon_section.get("bounded_execute_resumed"))
        ),
        "artifact_judgment_readback": (
            int(recursive_daemon_section.get("artifact_readback_auto_judgment_set_count") or 0) >= 1
            and int(recursive_daemon_section.get("artifact_readback_accept_count") or 0) >= 1
            and bool(recursive_daemon_section.get("artifact_readback_resumed"))
        ),
        "real_controlled_readback": (
            int(recursive_daemon_section.get("real_artifact_readback_judgment_set_count") or 0) >= 1
            and int(recursive_daemon_section.get("real_artifact_readback_trigger_judgment_count") or 0) >= 3
            and int(recursive_daemon_section.get("real_artifact_readback_control_judgment_count") or 0) >= 1
            and int(recursive_daemon_section.get("real_artifact_readback_control_loss_count") or 0) == 0
            and int(recursive_daemon_section.get("real_artifact_readback_accept_count") or 0) >= 1
            and bool(recursive_daemon_section.get("real_artifact_readback_resumed"))
        ),
        "gated_retention_and_no_ungated_mutation": (
            int(recursive_daemon_section.get("accepted_apply_count") or 0) >= 2
            and int(recursive_daemon_section.get("bounded_execute_applied_count") or 0) == 0
            and int(recursive_daemon_section.get("real_artifact_readback_applied_count") or 0) == 0
        ),
    }
    gates = [
        {
            "gate": name,
            "pass": passed,
            "observed": _gate_observed(name, recursive_daemon_section),
        }
        for name, passed in gate_map.items()
    ]
    pass_condition = (
        len(selected_cycles) >= 5
        and all(gate["pass"] for gate in gates)
        and _ungated_mutation_count(recursive_daemon_section) == 0
    )
    return {
        "eval_id": eval_id,
        "eval_kind": "budgeted_continuous_daemon_autonomy_audit",
        "pass": pass_condition,
        "budgeted_continuous_mode": True,
        "continuous_background_mode": False,
        "graph_mutation_policy": "gated_apply_only",
        "ungated_graph_mutation_count": _ungated_mutation_count(recursive_daemon_section),
        "cycle_count": len(selected_cycles),
        "cycles": selected_cycles,
        "gates": gates,
        "summary": {
            "preflight_queue_ready_count": recursive_daemon_section.get("preflight_queue_ready_count"),
            "preflight_queue_consumed": recursive_daemon_section.get("preflight_queue_consumed"),
            "bounded_execute_succeeded_leaf_count": recursive_daemon_section.get("bounded_execute_succeeded_leaf_count"),
            "bounded_execute_resumed": recursive_daemon_section.get("bounded_execute_resumed"),
            "artifact_readback_auto_judgment_set_count": recursive_daemon_section.get("artifact_readback_auto_judgment_set_count"),
            "artifact_readback_accept_count": recursive_daemon_section.get("artifact_readback_accept_count"),
            "artifact_readback_resumed": recursive_daemon_section.get("artifact_readback_resumed"),
            "real_artifact_readback_trigger_judgment_count": recursive_daemon_section.get("real_artifact_readback_trigger_judgment_count"),
            "real_artifact_readback_control_judgment_count": recursive_daemon_section.get("real_artifact_readback_control_judgment_count"),
            "real_artifact_readback_control_loss_count": recursive_daemon_section.get("real_artifact_readback_control_loss_count"),
            "real_artifact_readback_accept_count": recursive_daemon_section.get("real_artifact_readback_accept_count"),
            "accepted_apply_count": recursive_daemon_section.get("accepted_apply_count"),
            "manifest_backed": int(recursive_daemon_section.get("preflight_queue_manifest_count") or 0) >= 5,
        },
        "scope_note": (
            "This audit proves a bounded autonomous daemon loop, not an always-on background service. "
            "It can consume a frontier queue, read generated or cached judgments, resume recursion, "
            "and apply only gated accepted candidates."
        ),
    }


def _cycles(section: dict) -> list[dict]:
    return [
        {
            "cycle": 1,
            "name": "frontier_queue_discovery",
            "status": "passed" if section.get("preflight_queue_consumed") else "failed",
            "evidence": {
                "ready_count": section.get("preflight_queue_ready_count"),
                "planned_leaf_count": section.get("preflight_queue_planned_leaf_count"),
                "executable_leaf_count": section.get("preflight_queue_executable_leaf_count"),
                "manifest_count": section.get("preflight_queue_manifest_count"),
            },
            "next_transition": "execute_or_artifact_readback",
        },
        {
            "cycle": 2,
            "name": "bounded_execute_read_resume",
            "status": "passed" if section.get("bounded_execute_resumed") else "failed",
            "evidence": {
                "succeeded_leaf_count": section.get("bounded_execute_succeeded_leaf_count"),
                "accept_count": section.get("bounded_execute_accept_count"),
                "status_counts": section.get("bounded_execute_status_counts"),
                "applied_count": section.get("bounded_execute_applied_count"),
            },
            "next_transition": "candidate_acceptance_resume",
        },
        {
            "cycle": 3,
            "name": "artifact_judgment_readback",
            "status": "passed" if section.get("artifact_readback_resumed") else "failed",
            "evidence": {
                "auto_judgment_set_count": section.get("artifact_readback_auto_judgment_set_count"),
                "accept_count": section.get("artifact_readback_accept_count"),
                "resumed": section.get("artifact_readback_resumed"),
            },
            "next_transition": "controlled_readback_validation",
        },
        {
            "cycle": 4,
            "name": "real_controlled_readback",
            "status": "passed" if section.get("real_artifact_readback_resumed") else "failed",
            "evidence": {
                "judgment_set_count": section.get("real_artifact_readback_judgment_set_count"),
                "trigger_judgment_count": section.get("real_artifact_readback_trigger_judgment_count"),
                "control_judgment_count": section.get("real_artifact_readback_control_judgment_count"),
                "control_loss_count": section.get("real_artifact_readback_control_loss_count"),
                "accept_count": section.get("real_artifact_readback_accept_count"),
            },
            "next_transition": "gated_retention",
        },
        {
            "cycle": 5,
            "name": "gated_retention_policy",
            "status": "passed" if int(section.get("accepted_apply_count") or 0) >= 2 else "failed",
            "evidence": {
                "accepted_apply_count": section.get("accepted_apply_count"),
                "bounded_execute_applied_count": section.get("bounded_execute_applied_count"),
                "real_artifact_readback_applied_count": section.get("real_artifact_readback_applied_count"),
                "ungated_graph_mutation_count": _ungated_mutation_count(section),
            },
            "next_transition": "next_frontier_or_stop_budget",
        },
    ]


def _ungated_mutation_count(section: dict) -> int:
    return max(0, int(section.get("bounded_execute_applied_count") or 0)) + max(
        0,
        int(section.get("real_artifact_readback_applied_count") or 0),
    )


def _gate_observed(name: str, section: dict) -> dict:
    keys = {
        "frontier_queue_coverage": (
            "preflight_queue_consumed",
            "preflight_queue_ready_count",
            "preflight_queue_planned_leaf_count",
            "preflight_queue_executable_leaf_count",
        ),
        "bounded_execute_read_resume": (
            "bounded_execute_succeeded_leaf_count",
            "bounded_execute_accept_count",
            "bounded_execute_resumed",
            "bounded_execute_applied_count",
        ),
        "artifact_judgment_readback": (
            "artifact_readback_auto_judgment_set_count",
            "artifact_readback_accept_count",
            "artifact_readback_resumed",
        ),
        "real_controlled_readback": (
            "real_artifact_readback_judgment_set_count",
            "real_artifact_readback_trigger_judgment_count",
            "real_artifact_readback_control_judgment_count",
            "real_artifact_readback_control_loss_count",
            "real_artifact_readback_accept_count",
            "real_artifact_readback_resumed",
        ),
        "gated_retention_and_no_ungated_mutation": (
            "accepted_apply_count",
            "bounded_execute_applied_count",
            "real_artifact_readback_applied_count",
        ),
    }
    return {key: section.get(key) for key in keys.get(name, ())}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--performance-payload", default="phase four/assumption_graph/reconstruction_gap_perf_20260602_external_v5_objective.json")
    ap.add_argument("--eval-id", default="continuous_daemon_autonomy_20260604")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    perf = _load_json(_resolve(root, args.performance_payload))
    sections = perf.get("sections", perf)
    payload = build_continuous_daemon_autonomy_payload(
        eval_id=args.eval_id,
        recursive_daemon_section=sections.get("recursive_daemon", {}),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out = _resolve(root, args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
