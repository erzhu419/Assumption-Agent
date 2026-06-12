"""Claim-frontier advancement beyond the current paper evidence pack.

The prior evidence pack correctly blocks three overclaims:
unbounded 24/7 autonomy, simulator-as-judge replacement, and a full
category-theory theorem prover.  This module advances the frontier without
weakening those boundaries: it promotes only the next bounded claims that are
supported by the existing production-candidate artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_OUT = PAPER_DIR / "claim_frontier_advancement_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/claim_frontier_advancement_20260612.md")

SOURCE_ARTIFACTS = {
    "self_evo_paper_pack": PAPER_DIR / "self_evo_paper_evidence_pack_20260612.json",
    "autonomy": PAPER_DIR / "autonomy_supervised_production_run_20260612.json",
    "simulator_gate": PAPER_DIR / "simulator_production_gate_20260612.json",
    "simulator_evidence": PAPER_DIR / "simulator_production_evidence_20260612.json",
    "formal_stack": PAPER_DIR / "finite_formal_reasoning_stack_20260612.json",
    "integrated_episode": PAPER_DIR / "integrated_recursive_episode_b3_c2_20260612.json",
}


def build_claim_frontier_advancement_payload(
    *,
    root: Path,
    eval_id: str = "claim_frontier_advancement_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    metrics = _metrics(artifacts)
    tracks = _frontier_tracks(metrics)
    allowed_claims = [
        row["next_bounded_claim"]
        for row in tracks
        if row["frontier_l3p5_allowed"]
    ]
    blocked_claims = _blocked_claims(metrics)
    gates = {
        "all_source_artifacts_present": all((root / path).exists() for path in SOURCE_ARTIFACTS.values()),
        "all_source_artifacts_pass": metrics["source_artifact_pass_rate"] == 1.0,
        "autonomy_frontier_l3p5_allowed": tracks[0]["frontier_l3p5_allowed"] is True,
        "simulator_frontier_l3p5_allowed": tracks[1]["frontier_l3p5_allowed"] is True,
        "formal_frontier_l3p5_allowed": tracks[2]["frontier_l3p5_allowed"] is True,
        "blocked_claims_still_blocked": all(row["allowed"] is False for row in blocked_claims),
        "frontier_score_high": metrics["frontier_advancement_score"] >= 0.90,
        "next_evidence_requirements_recorded": all(row["next_evidence_required"] for row in tracks),
        "no_main_graph_or_policy_default_overclaim": metrics["main_graph_mutation_count"] == 0
        and metrics["ungated_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "claim_frontier_advancement",
        "performance_validation": True,
        "validation_scope": (
            "Advances the paper claim frontier from L3 production-candidate evidence to L3.5 bounded next claims "
            "for supervised autonomy, simulator triage, and finite formal transfer.  It deliberately keeps "
            "unbounded autonomy, simulator-as-judge replacement, and full theorem-prover claims blocked."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass")),
                "eval_kind": artifacts[name].get("eval_kind"),
                "sha256": _sha256(root / path) if (root / path).exists() else None,
            }
            for name, path in SOURCE_ARTIFACTS.items()
        },
        "frontier_tracks": tracks,
        "allowed_bounded_next_claims": allowed_claims,
        "blocked_claims": blocked_claims,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Claim Frontier Advancement",
        "",
        f"- pass: `{payload['pass']}`",
        f"- frontier advancement score: `{m['frontier_advancement_score']}`",
        f"- L3.5 tracks: `{m['frontier_track_pass_count']}/{m['frontier_track_count']}`",
        f"- source artifact pass rate: `{m['source_artifact_pass_rate']}`",
        f"- blocked overclaim count: `{m['blocked_overclaim_count']}`",
        "",
        "## Frontier Tracks",
        "",
        "| Track | Achieved | Next bounded claim | Score | Allowed | Still blocked |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["frontier_tracks"]:
        lines.append(
            f"| `{row['track_id']}` | {row['achieved_claim']} | {row['next_bounded_claim']} | "
            f"`{row['frontier_score']}` | `{row['frontier_l3p5_allowed']}` | {row['blocked_upper_claim']} |"
        )
    lines.extend(["", "## Bounded Claims Now Supported", ""])
    for claim in payload["allowed_bounded_next_claims"]:
        lines.append(f"- {claim}")
    lines.extend(["", "## Claims Still Blocked", ""])
    for row in payload["blocked_claims"]:
        lines.append(f"- `{row['claim_id']}`: allowed=`{row['allowed']}`; {row['reason']}")
    lines.extend(["", "## Next Evidence Required", ""])
    for row in payload["frontier_tracks"]:
        lines.append(f"- `{row['track_id']}`: {row['next_evidence_required']}")
    return "\n".join(lines).rstrip() + "\n"


def _metrics(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    autonomy = artifacts["autonomy"].get("metrics", {})
    sim_gate = artifacts["simulator_gate"].get("metrics", {})
    sim_evidence = artifacts["simulator_evidence"].get("metrics", {})
    formal = artifacts["formal_stack"].get("metrics", {})
    integrated = artifacts["integrated_episode"].get("metrics", {})
    paper = artifacts["self_evo_paper_pack"].get("metrics", {})
    pass_rate = sum(1 for payload in artifacts.values() if payload.get("pass")) / max(1, len(artifacts))

    autonomy_score = _bounded_mean(
        _cap(float(autonomy.get("supervised_day_count", 0)) / 30.0),
        _cap(float(autonomy.get("cycle_count", 0)) / 720.0),
        float(autonomy.get("low_risk_auto_apply_precision", 0.0)),
        1.0 - float(autonomy.get("downstream_regression_rate", 1.0)),
        1.0 - float(autonomy.get("human_override_rate", 1.0)),
        1.0 if autonomy.get("all_applies_replayable") else 0.0,
        1.0 if int(autonomy.get("forbidden_policy_change_auto_apply_count", 1)) == 0 else 0.0,
    )
    simulator_score = _bounded_mean(
        _cap(float(sim_gate.get("transition_row_count", 0)) / 2000.0),
        _cap(float(sim_gate.get("pattern_count", 0)) / 20.0),
        _cap(float(sim_evidence.get("same_state_group_count", 0)) / 180.0),
        1.0 if sim_gate.get("no_leakage_audit_pass") else 0.0,
        1.0 - _cap(float(sim_evidence.get("counterfactual_mae", 1.0)) / 0.05),
        float(sim_evidence.get("best_arm_agreement_rate", 0.0)),
        _cap(float(sim_evidence.get("policy_lift_over_v3", 0.0)) / 0.25),
    )
    formal_score = _bounded_mean(
        _cap(float(formal.get("finite_theorem_fragment_external_lean_theorem_count", 0)) / 30.0),
        _cap(float(formal.get("certificate_count", 0)) / 16.0),
        float(formal.get("finite_theorem_fragment_nl_certificate_pass_rate", 0.0)),
        float(formal.get("formal_transfer_negative_control_rejection_rate", 0.0)),
        1.0 if formal.get("lean_verified_finite_theorem_fragment_claim_allowed") else 0.0,
        1.0 if not formal.get("full_theorem_prover_claim_allowed") else 0.0,
    )
    frontier_score = round((autonomy_score + simulator_score + formal_score) / 3.0, 4)
    return {
        "source_artifact_count": len(artifacts),
        "source_artifact_pass_rate": round(pass_rate, 4),
        "autonomy_frontier_score": autonomy_score,
        "autonomy_supervised_day_count": int(autonomy.get("supervised_day_count", 0)),
        "autonomy_cycle_count": int(autonomy.get("cycle_count", 0)),
        "autonomy_auto_apply_count": int(autonomy.get("auto_apply_count", 0)),
        "autonomy_low_risk_auto_apply_precision": float(autonomy.get("low_risk_auto_apply_precision", 0.0)),
        "autonomy_downstream_regression_rate": float(autonomy.get("downstream_regression_rate", 1.0)),
        "autonomy_human_override_rate": float(autonomy.get("human_override_rate", 1.0)),
        "production_autonomy_candidate_allowed": bool(autonomy.get("production_autonomy_candidate_allowed")),
        "ungated_mutation_count": int(autonomy.get("ungated_mutation_count", 0)),
        "simulator_frontier_score": simulator_score,
        "simulator_transition_row_count": int(sim_gate.get("transition_row_count", 0)),
        "simulator_pattern_count": int(sim_gate.get("pattern_count", 0)),
        "simulator_same_state_group_count": int(sim_evidence.get("same_state_group_count", 0)),
        "simulator_counterfactual_mae": float(sim_evidence.get("counterfactual_mae", 1.0)),
        "simulator_global_baseline_mae": float(sim_evidence.get("global_baseline_mae", 0.0)),
        "simulator_best_arm_agreement_rate": float(sim_evidence.get("best_arm_agreement_rate", 0.0)),
        "simulator_policy_lift_over_v3": float(sim_evidence.get("policy_lift_over_v3", 0.0)),
        "production_simulator_candidate_allowed": bool(sim_gate.get("production_simulator_candidate_allowed")),
        "simulator_state_feature_leak_count": int(sim_gate.get("state_feature_leak_count", 1)),
        "simulator_prediction_outcome_exact_identity_count": int(
            sim_gate.get("prediction_outcome_exact_identity_count", 1)
        ),
        "formal_frontier_score": formal_score,
        "formal_lean_theorem_count": int(formal.get("finite_theorem_fragment_external_lean_theorem_count", 0)),
        "formal_certificate_count": int(formal.get("certificate_count", 0)),
        "formal_nl_certificate_pass_rate": float(
            formal.get("finite_theorem_fragment_nl_certificate_pass_rate", 0.0)
        ),
        "formal_negative_control_rejection_rate": float(
            formal.get("formal_transfer_negative_control_rejection_rate", 0.0)
        ),
        "lean_verified_finite_theorem_fragment_claim_allowed": bool(
            formal.get("lean_verified_finite_theorem_fragment_claim_allowed")
        ),
        "full_theorem_prover_claim_allowed": bool(formal.get("full_theorem_prover_claim_allowed")),
        "integrated_episode_pass": bool(artifacts["integrated_episode"].get("pass")),
        "integrated_fresh_ablation_accept_count": int(integrated.get("fresh_ablation_accept_count", 0)),
        "integrated_fresh_ablation_reject_count": int(integrated.get("fresh_ablation_reject_count", 0)),
        "integrated_abstained_auto_execute_count": int(integrated.get("abstained_candidate_auto_execute_count", 1)),
        "main_graph_mutation_count": int(integrated.get("main_graph_mutation_count", 1)),
        "paper_pack_roadmap_open_item_count": int(paper.get("roadmap_open_item_count", 1)),
        "frontier_advancement_score": frontier_score,
        "frontier_track_count": 3,
        "frontier_track_pass_count": 0,
        "blocked_overclaim_count": 3,
        "evidence_hash": stable_hash({
            "autonomy": autonomy,
            "simulator_gate": sim_gate,
            "simulator_evidence": sim_evidence,
            "formal": formal,
            "integrated": integrated,
        }),
    }


def _frontier_tracks(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = [
        {
            "track_id": "A_autonomy",
            "achieved_claim": "L3 restricted supervised production autonomy candidate",
            "next_bounded_claim": (
                "L3.5 replayable supervised low-risk autonomy with 30-day-equivalent evidence, "
                "zero downstream regression, and manual escalation for policy/default/formal mutations"
            ),
            "blocked_upper_claim": "L4 unbounded 24/7 autonomous self-evolution OS",
            "frontier_score": metrics["autonomy_frontier_score"],
            "frontier_l3p5_allowed": (
                metrics["production_autonomy_candidate_allowed"] is True
                and metrics["autonomy_supervised_day_count"] >= 30
                and metrics["autonomy_cycle_count"] >= 720
                and metrics["autonomy_low_risk_auto_apply_precision"] >= 0.99
                and metrics["autonomy_downstream_regression_rate"] == 0.0
                and metrics["ungated_mutation_count"] == 0
            ),
            "next_evidence_required": (
                "real wall-clock multi-week service logs, multi-project deployment, incident reports, "
                "budget/rate-limit monitors, and human override audits"
            ),
        },
        {
            "track_id": "B_simulator",
            "achieved_claim": "L3 production graph-action simulator for triage and verifier routing",
            "next_bounded_claim": (
                "L3.5 selective simulator deferral for low-risk graph-maintenance decisions with audit sampling; "
                "live ablation and judges remain required for promotion claims"
            ),
            "blocked_upper_claim": "L4 world simulator replacing live validation or judges",
            "frontier_score": metrics["simulator_frontier_score"],
            "frontier_l3p5_allowed": (
                metrics["production_simulator_candidate_allowed"] is True
                and metrics["simulator_transition_row_count"] >= 2000
                and metrics["simulator_pattern_count"] >= 20
                and metrics["simulator_counterfactual_mae"] <= 0.01
                and metrics["simulator_best_arm_agreement_rate"] >= 0.95
                and metrics["simulator_policy_lift_over_v3"] >= 0.20
                and metrics["simulator_state_feature_leak_count"] == 0
                and metrics["simulator_prediction_outcome_exact_identity_count"] == 0
            ),
            "next_evidence_required": (
                "fresh same-state multi-arm live rows across more domains, prospective audit-sampling logs, "
                "and calibration curves under distribution shift"
            ),
        },
        {
            "track_id": "C_formal",
            "achieved_claim": "L3 Lean-verified finite theorem fragment for bounded formal mappings",
            "next_bounded_claim": (
                "L3.5 proof-carrying finite transfer kernel: every promoted morphism supplies a finite diagram, "
                "negative controls, and an external Lean-checked theorem-fragment certificate"
            ),
            "blocked_upper_claim": "L4 full category-theory theorem prover",
            "frontier_score": metrics["formal_frontier_score"],
            "frontier_l3p5_allowed": (
                metrics["lean_verified_finite_theorem_fragment_claim_allowed"] is True
                and metrics["formal_lean_theorem_count"] >= 30
                and metrics["formal_certificate_count"] >= 16
                and metrics["formal_nl_certificate_pass_rate"] == 1.0
                and metrics["formal_negative_control_rejection_rate"] == 1.0
                and metrics["full_theorem_prover_claim_allowed"] is False
            ),
            "next_evidence_required": (
                "larger NL-to-diagram benchmark, proof-carrying graph writeback for promoted morphisms, "
                "and external proof-assistant dependency manifests"
            ),
        },
    ]
    pass_count = sum(1 for row in tracks if row["frontier_l3p5_allowed"])
    metrics["frontier_track_pass_count"] = pass_count
    return tracks


def _blocked_claims(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "unbounded_24_7_autonomous_self_evolution_os",
            "allowed": False,
            "reason": (
                "Current evidence supports supervised restricted low-risk autonomy only; "
                f"main_graph_mutation_count={metrics['main_graph_mutation_count']} and "
                f"ungated_mutation_count={metrics['ungated_mutation_count']}."
            ),
        },
        {
            "claim_id": "world_simulator_replacing_live_ablation_or_judges",
            "allowed": False,
            "reason": (
                "Simulator evidence supports triage, routing, and selective deferral only; "
                f"counterfactual_mae={metrics['simulator_counterfactual_mae']} still requires audit sampling."
            ),
        },
        {
            "claim_id": "full_category_theory_theorem_prover",
            "allowed": False,
            "reason": (
                "Formal evidence is Lean-verified for a finite theorem fragment only; "
                f"finite theorem count={metrics['formal_lean_theorem_count']}."
            ),
        },
    ]


def _bounded_mean(*values: float) -> float:
    return round(sum(_cap(value) for value in values) / max(1, len(values)), 4)


def _cap(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--eval-id", default="claim_frontier_advancement_20260612")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    payload = build_claim_frontier_advancement_payload(root=args.root, eval_id=args.eval_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "failed_gates": payload["failed_gates"],
                "metrics": payload["metrics"],
                "out": str(args.out.resolve()),
                "md_out": str(args.md_out.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
