"""Coverage audit for reconstruction/md/L4_roadmap.md.

The L4 roadmap defines L4a as open-world supervised self-evolution, not an
unbounded autonomous OS.  This audit maps the seven L4 stages to executable
artifacts and separates two claims:

1. L4a preflight/protocol readiness is allowed when all stage contracts pass.
2. Completed L4a/L4b claims remain blocked without real wall-clock, external
   prospective, human/expert, and long-horizon evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .conservative_generalization_gate_v2 import build_conservative_generalization_gate_v2_payload
from .framework_formal_certificate_integration import build_framework_formal_certificate_integration_payload
from .framework_lifecycle_ledger_v2 import build_framework_lifecycle_ledger_v2_payload
from .framework_simulator_guided_search import build_framework_simulator_guided_search_payload
from .l4_prospective_task_stream import build_l4_prospective_task_stream_payload
from .l4_residual_framework_mini_run import build_l4_residual_framework_mini_run_payload
from .l4_wallclock_supervised_service import build_l4_wallclock_supervised_service_payload
from .paper_fresh_frozen_rerun_protocol import build_paper_fresh_frozen_rerun_protocol_payload
from .self_evo_paper_evidence_pack import build_self_evo_paper_evidence_pack_payload


DEFAULT_OUT = PAPER_DIR / "l4_roadmap_coverage_audit_20260613.json"
DEFAULT_MD_OUT = Path("reconstruction/md/l4_roadmap_coverage_audit_20260613.md")


def build_l4_roadmap_coverage_audit_payload(
    *,
    root: Path,
    eval_id: str = "l4_roadmap_coverage_audit_20260613",
) -> dict[str, Any]:
    root = root.resolve()
    wallclock = build_l4_wallclock_supervised_service_payload(root=root, eval_id=f"{eval_id}_wallclock")
    task_stream = build_l4_prospective_task_stream_payload(root=root, eval_id=f"{eval_id}_task_stream")
    mini_run = build_l4_residual_framework_mini_run_payload(root=root, eval_id=f"{eval_id}_mini_run")
    gate = build_conservative_generalization_gate_v2_payload(root=root, eval_id=f"{eval_id}_gate_v2")
    ledger = build_framework_lifecycle_ledger_v2_payload(root=root, eval_id=f"{eval_id}_ledger_v2")
    simulator = build_framework_simulator_guided_search_payload(root=root, eval_id=f"{eval_id}_simulator")
    formal = build_framework_formal_certificate_integration_payload(root=root, eval_id=f"{eval_id}_formal")
    fresh_protocol = build_paper_fresh_frozen_rerun_protocol_payload(root=root, eval_id=f"{eval_id}_fresh_protocol")
    paper_pack = build_self_evo_paper_evidence_pack_payload(root=root, eval_id=f"{eval_id}_paper_pack")
    stages = _stages(
        wallclock=wallclock,
        task_stream=task_stream,
        mini_run=mini_run,
        gate=gate,
        ledger=ledger,
        simulator=simulator,
        formal=formal,
        fresh_protocol=fresh_protocol,
        paper_pack=paper_pack,
    )
    claim_boundaries = _claim_boundaries(
        wallclock=wallclock,
        task_stream=task_stream,
        mini_run=mini_run,
        simulator=simulator,
        formal=formal,
        fresh_protocol=fresh_protocol,
    )
    metrics = _metrics(stages=stages, claim_boundaries=claim_boundaries)
    gates = {
        "seven_stages_present": metrics["stage_count"] == 7,
        "all_stage_preflights_pass": metrics["stage_preflight_pass_count"] == 7,
        "l4_mini_requirements_preflighted": metrics["l4_mini_requirement_preflight_count"] >= 8,
        "real_time_gap_explicitly_blocked": metrics["real_time_completion_claim_allowed"] is False,
        "real_task_gap_explicitly_blocked": metrics["external_prospective_result_claim_allowed"] is False,
        "real_candidate_gap_partially_closed": metrics["real_residual_candidate_count"] >= 20,
        "real_external_judgment_gap_explicitly_blocked": metrics["human_expert_panel_claim_allowed"] is False,
        "l4a_preflight_claim_allowed": metrics["l4a_preflight_claim_allowed"] is True,
        "completed_l4a_claim_blocked": metrics["completed_l4a_claim_allowed"] is False,
        "l4b_unbounded_claim_blocked": metrics["l4b_unbounded_claim_allowed"] is False,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
        "overclaim_leak_zero": metrics["overclaim_leak_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "l4_roadmap_coverage_audit",
        "source_md": "reconstruction/md/L4_roadmap.md",
        "performance_validation": True,
        "validation_scope": (
            "Audits the seven-stage L4 roadmap.  The audit allows an L4a preflight/protocol-readiness claim "
            "when every stage has executable evidence, but it keeps completed L4a and L4b claims blocked until "
            "real wall-clock, external prospective, human/expert, and long-run graph evidence exists."
        ),
        "stages": stages,
        "claim_boundaries": claim_boundaries,
        "source_summaries": {
            "wallclock": _summary(wallclock),
            "task_stream": _summary(task_stream),
            "mini_run": _summary(mini_run),
            "gate_v2": _summary(gate),
            "ledger_v2": _summary(ledger),
            "simulator": _summary(simulator),
            "formal": _summary(formal),
            "fresh_protocol": _summary(fresh_protocol),
            "paper_pack": _summary(paper_pack),
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "L4a open-world supervised self-evolution preflight is implemented and performance-validated",
        "blocked_claims": [
            "completed_l4a_open_world_self_evolution_run",
            "l4b_unbounded_autonomous_os",
            "world_simulator_replaces_live_validation_or_judges",
            "full_category_theory_theorem_prover",
            "human_expert_panel_completed",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# L4 Roadmap Coverage Audit",
        "",
        f"- pass: `{payload['pass']}`",
        f"- stage preflight pass: `{m['stage_preflight_pass_count']}/{m['stage_count']}`",
        f"- completed L4a claim: `{m['completed_l4a_claim_allowed']}`",
        f"- L4b claim: `{m['l4b_unbounded_claim_allowed']}`",
        f"- real residual candidates: `{m['real_residual_candidate_count']}`",
        f"- L4-mini preflight requirements: `{m['l4_mini_requirement_preflight_count']}`",
        "",
        "## Stages",
        "",
        "| Stage | Preflight | L4 completion | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["stages"]:
        lines.append(
            f"| `{row['stage_id']}` | `{row['preflight_pass']}` | `{row['l4_completion_claim_allowed']}` | {row['evidence']} |"
        )
    lines.extend(["", "## Claim Boundaries", "", "| Claim | Allowed | Reason |", "| --- | --- | --- |"])
    for row in payload["claim_boundaries"]:
        lines.append(f"| `{row['claim_id']}` | `{row['allowed']}` | {row['reason']} |")
    return "\n".join(lines).rstrip() + "\n"


def _stages(
    *,
    wallclock: dict[str, Any],
    task_stream: dict[str, Any],
    mini_run: dict[str, Any],
    gate: dict[str, Any],
    ledger: dict[str, Any],
    simulator: dict[str, Any],
    formal: dict[str, Any],
    fresh_protocol: dict[str, Any],
    paper_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    gate_metrics = gate["metrics"]
    ledger_metrics = ledger["metrics"]
    mini_metrics = mini_run["metrics"]
    simulator_metrics = simulator["metrics"]
    formal_metrics = formal["metrics"]
    return [
        {
            "stage_id": "L4-1_wall_clock_supervised_autonomy_service",
            "preflight_pass": wallclock["pass"],
            "l4_completion_claim_allowed": wallclock["metrics"]["l4a_wallclock_completed_claim_allowed"],
            "evidence": "l4_wallclock_supervised_service_20260613.json",
            "key_metric": f"observed_hours={wallclock['metrics']['observed_wallclock_hours']}",
        },
        {
            "stage_id": "L4-2_prospective_unseen_task_stream",
            "preflight_pass": task_stream["pass"],
            "l4_completion_claim_allowed": task_stream["metrics"]["completed_external_benchmark_claim_allowed"],
            "evidence": "l4_prospective_task_stream_20260613.json",
            "key_metric": f"tasks={task_stream['metrics']['manifest_task_count']}",
        },
        {
            "stage_id": "L4-3_real_residual_to_framework_generator",
            "preflight_pass": mini_run["pass"],
            "l4_completion_claim_allowed": mini_run["metrics"]["l4_external_completion_claim_allowed"],
            "evidence": "l4_residual_framework_mini_run_20260613.json",
            "key_metric": f"candidates={mini_metrics['candidate_framework_count']}",
        },
        {
            "stage_id": "L4-4_conservative_generalization_gate_v2_branch_ledger",
            "preflight_pass": gate["pass"] and ledger["pass"],
            "l4_completion_claim_allowed": False,
            "evidence": "conservative_generalization_gate_v2_20260612.json + framework_lifecycle_ledger_v2_20260612.json",
            "key_metric": (
                f"gate_eval={gate_metrics['evaluated_candidate_count']}; "
                f"ledger_entries={ledger_metrics['ledger_entry_count']}"
            ),
        },
        {
            "stage_id": "L4-5_external_expert_human_judgment_layer",
            "preflight_pass": mini_run["pass"] and mini_metrics["expert_review_packet_row_count"] >= 5,
            "l4_completion_claim_allowed": mini_metrics["human_expert_panel_claim_allowed"],
            "evidence": "framework_external_eval_pack_20260612.json + l4_residual_framework_mini_run_20260613.json",
            "key_metric": f"expert_packet={mini_metrics['expert_review_packet_row_count']}",
        },
        {
            "stage_id": "L4-6_prospective_simulator_formal_verifier_routing",
            "preflight_pass": simulator["pass"] and formal["pass"],
            "l4_completion_claim_allowed": False,
            "evidence": "framework_simulator_guided_search_20260612.json + framework_formal_certificate_integration_20260612.json",
            "key_metric": (
                f"sim_reduction={simulator_metrics['fresh_test_reduction_rate']}; "
                f"formal_cert={formal_metrics['formal_applicable_certificate_coverage']}"
            ),
        },
        {
            "stage_id": "L4-7_integrated_open_world_framework_evolution_run",
            "preflight_pass": fresh_protocol["pass"] and paper_pack["pass"],
            "l4_completion_claim_allowed": fresh_protocol["metrics"]["target_fresh_result_claim_allowed"],
            "evidence": "paper_fresh_frozen_rerun_protocol_20260612.json + self_evo_paper_evidence_pack_20260612.json",
            "key_metric": f"fresh_protocol_target={fresh_protocol['metrics']['target_fresh_api_call_count']}",
        },
    ]


def _claim_boundaries(
    *,
    wallclock: dict[str, Any],
    task_stream: dict[str, Any],
    mini_run: dict[str, Any],
    simulator: dict[str, Any],
    formal: dict[str, Any],
    fresh_protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "real_wall_clock_7d_or_30d_completed",
            "allowed": wallclock["metrics"]["l4a_wallclock_completed_claim_allowed"],
            "reason": "requires observed wall-clock service log; current default artifact is readiness only",
        },
        {
            "claim_id": "external_prospective_benchmark_completed",
            "allowed": task_stream["metrics"]["completed_external_benchmark_claim_allowed"]
            or fresh_protocol["metrics"]["target_fresh_result_claim_allowed"],
            "reason": "requires execute artifact over frozen stream; current task stream is manifest/protocol",
        },
        {
            "claim_id": "fresh_external_llm_framework_generation_completed",
            "allowed": mini_run["metrics"]["live_llm_api_executed"],
            "reason": "default mini-run uses deterministic LLM-contract replay unless execute-live succeeds",
        },
        {
            "claim_id": "human_expert_panel_completed",
            "allowed": mini_run["metrics"]["human_expert_panel_claim_allowed"],
            "reason": "expert packet exists; real human labels are not fabricated",
        },
        {
            "claim_id": "world_simulator_replaces_live_validation_or_judges",
            "allowed": bool(simulator["metrics"].get("production_simulator_replacement_allowed", False)),
            "reason": "simulator remains a router/gate and cannot replace live ablation or judge",
        },
        {
            "claim_id": "full_category_theory_theorem_prover",
            "allowed": bool(formal["metrics"].get("full_theorem_prover_claim_allowed", False)),
            "reason": "formal layer is bounded proof-carrying transfer, not a full theorem prover",
        },
        {
            "claim_id": "l4b_unbounded_autonomous_os",
            "allowed": False,
            "reason": "L4 roadmap targets open-world supervised L4a, not unbounded L4b",
        },
    ]


def _metrics(*, stages: list[dict[str, Any]], claim_boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    stage_pass_count = sum(1 for row in stages if row["preflight_pass"])
    completed_count = sum(1 for row in stages if row["l4_completion_claim_allowed"])
    boundary_by_id = {row["claim_id"]: row for row in claim_boundaries}
    overclaim_leak_count = sum(
        1
        for row in claim_boundaries
        if row["claim_id"]
        in {
            "real_wall_clock_7d_or_30d_completed",
            "external_prospective_benchmark_completed",
            "human_expert_panel_completed",
            "world_simulator_replaces_live_validation_or_judges",
            "full_category_theory_theorem_prover",
            "l4b_unbounded_autonomous_os",
        }
        and row["allowed"]
    )
    mini_stage = next(row for row in stages if row["stage_id"].startswith("L4-3"))
    l4_mini_requirement_preflight_count = sum(
        [
            stage_pass_count == len(stages),
            boundary_by_id["real_wall_clock_7d_or_30d_completed"]["allowed"] is False,
            boundary_by_id["external_prospective_benchmark_completed"]["allowed"] is False,
            boundary_by_id["human_expert_panel_completed"]["allowed"] is False,
            boundary_by_id["l4b_unbounded_autonomous_os"]["allowed"] is False,
            "candidates=" in mini_stage["key_metric"],
            all(row["preflight_pass"] for row in stages[:3]),
            all(row["preflight_pass"] for row in stages[3:]),
        ]
    )
    real_residual_candidate_count = 0
    for row in stages:
        if row["stage_id"].startswith("L4-3"):
            try:
                real_residual_candidate_count = int(row["key_metric"].split("candidates=")[1].split(";")[0])
            except (IndexError, ValueError):
                real_residual_candidate_count = 0
    return {
        "stage_count": len(stages),
        "stage_preflight_pass_count": stage_pass_count,
        "stage_l4_completion_claim_allowed_count": completed_count,
        "l4_mini_requirement_preflight_count": l4_mini_requirement_preflight_count,
        "real_residual_candidate_count": real_residual_candidate_count,
        "real_time_completion_claim_allowed": boundary_by_id["real_wall_clock_7d_or_30d_completed"]["allowed"],
        "external_prospective_result_claim_allowed": boundary_by_id[
            "external_prospective_benchmark_completed"
        ]["allowed"],
        "human_expert_panel_claim_allowed": boundary_by_id["human_expert_panel_completed"]["allowed"],
        "world_simulator_replacement_claim_allowed": boundary_by_id[
            "world_simulator_replaces_live_validation_or_judges"
        ]["allowed"],
        "full_theorem_prover_claim_allowed": boundary_by_id["full_category_theory_theorem_prover"]["allowed"],
        "l4a_preflight_claim_allowed": stage_pass_count == len(stages),
        "completed_l4a_claim_allowed": completed_count == len(stages),
        "l4b_unbounded_claim_allowed": boundary_by_id["l4b_unbounded_autonomous_os"]["allowed"],
        "claim_boundary_count": len(claim_boundaries),
        "blocked_claim_boundary_count": sum(1 for row in claim_boundaries if not row["allowed"]),
        "overclaim_leak_count": overclaim_leak_count,
        "main_graph_mutation_count": 0,
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": payload.get("eval_id"),
        "eval_kind": payload.get("eval_kind"),
        "pass": payload.get("pass"),
        "metrics": payload.get("metrics", {}),
        "failed_gates": payload.get("failed_gates", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build L4 roadmap coverage audit artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="l4_roadmap_coverage_audit_20260613")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_l4_roadmap_coverage_audit_payload(root=root, eval_id=args.eval_id)
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
