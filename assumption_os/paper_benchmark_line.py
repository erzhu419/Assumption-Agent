"""Paper-facing benchmark line for recursive assumption evolution.

This module turns the reconstruction target into one auditable line:

real tasks -> multiple hypotheses -> novelty/integration -> fresh ablation with
controls -> recursive resume -> gated apply/reject -> next-generation proposal.

It is intentionally stricter than the component-level performance validator.
Component tests can all pass while the paper claim remains under-supported; this
payload separates the working benchmark line from the remaining research gaps.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .morphism_benchmark import build_morphism_independent_benchmark_payload
from .novelty_integration import build_novelty_integration_performance_payload
from .recursive_evolution_proof import build_recursive_self_evolution_proof_payload
from .residual_diagnostics import build_large_residual_label_calibration_payload
from .formal_mapping import build_formal_engine_depth_payload
from .graph_memory import JsonlGraphStore
from .schema import stable_id


DEFAULT_PERFORMANCE_PATH = Path("phase four/assumption_graph/reconstruction_gap_perf_20260602_external_v5_objective.json")
DEFAULT_TRACE_DATASET_PATH = Path("phase four/assumption_graph/trace_dataset_collection_distilled_20260602.json")
DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/paper_benchmark_line_20260604.json")


def build_paper_benchmark_line_payload(
    *,
    root: Path,
    graph_dir: Path,
    eval_id: str | None = None,
    performance_payload: dict[str, Any] | None = None,
    performance_path: Path | None = None,
) -> dict[str, Any]:
    """Build a complete evidence line and explicit gap scoreboard."""

    eval_id = eval_id or "paper_benchmark_line_20260604"
    perf_path = performance_path or root / DEFAULT_PERFORMANCE_PATH
    performance_payload = performance_payload or _load_json(perf_path)
    sections = performance_payload.get("sections", {})
    recursive = build_recursive_self_evolution_proof_payload(eval_id=f"{eval_id}_recursive")
    morphism = build_morphism_independent_benchmark_payload(eval_id=f"{eval_id}_morphism")
    novelty = build_novelty_integration_performance_payload(eval_id=f"{eval_id}_novelty")
    graph_store = JsonlGraphStore(graph_dir)
    formal_depth = build_formal_engine_depth_payload(
        eval_id=f"{eval_id}_formal_engine_depth",
        store=graph_store,
        morphism_benchmark_payload=morphism,
    )
    trace_dataset_path = root / DEFAULT_TRACE_DATASET_PATH
    residual_calibration = build_large_residual_label_calibration_payload(
        eval_id=f"{eval_id}_large_residual_calibration",
        store=graph_store,
        trace_dataset_payload=_load_json(trace_dataset_path) if trace_dataset_path.exists() else None,
        target_examples=120,
    )

    line_gates = _benchmark_line_gates(
        sections=sections,
        recursive=recursive,
        morphism=morphism,
        novelty=novelty,
    )
    gap_gates = _research_gap_gates(
        sections=sections,
        recursive=recursive,
        morphism=morphism,
        novelty=novelty,
        residual_calibration=residual_calibration,
        formal_depth=formal_depth,
    )
    estimates = _completion_estimates(line_gates=line_gates, gap_gates=gap_gates)
    benchmark_line_pass = all(gate["pass"] for gate in line_gates)
    research_gap_pass = all(gate["pass"] for gate in gap_gates)
    return {
        "eval_id": eval_id,
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "performance_path": _display_path(root, perf_path),
            "performance_eval_id": performance_payload.get("eval_id"),
        },
        "thesis_line": (
            "real_task_set -> multi_hypothesis_generation -> novelty_integration -> "
            "fresh_ablation_with_controls -> recursive_resume -> gated_retention -> next_generation"
        ),
        "benchmark_line_pass": benchmark_line_pass,
        "research_gap_pass": research_gap_pass,
        "paper_readiness_pass": benchmark_line_pass and research_gap_pass,
        "benchmark_line_gates": line_gates,
        "research_gap_gates": gap_gates,
        "completion_estimates": estimates,
        "evidence_summaries": {
            "recursive_self_evolution": _recursive_summary(recursive),
            "morphism_benchmark": _morphism_summary(morphism),
            "novelty_integration": _novelty_summary(novelty),
            "large_residual_label_calibration": _large_residual_summary(residual_calibration),
            "formal_engine_depth": _formal_depth_summary(formal_depth),
            "performance_sections": _performance_summary(sections),
        },
        "next_actions_ranked": _next_actions(gap_gates),
    }


def _benchmark_line_gates(*, sections: dict[str, dict], recursive: dict, morphism: dict, novelty: dict) -> list[dict]:
    verifier = sections.get("verifier_stack", {})
    daemon = sections.get("recursive_daemon", {})
    surface = sections.get("surface_hypothesis_generator", {})
    trace_policy = sections.get("trace_policy_proposals", {})
    preflight = sections.get("trace_policy_preflight", {})
    trajectory = sections.get("trajectory_search", {})
    world = sections.get("world_model", {})
    trace_outcome = sections.get("trace_outcome_model", {})
    novelty_counts = novelty.get("classification_counts", {})
    mainline_cases = [
        gen.get("evidence", {}).get("selected_case_count", 0)
        for gen in recursive.get("mainline_generations", [])
    ]
    return [
        _gate(
            "real_task_multigeneration_trace",
            bool(recursive.get("pass"))
            and int(recursive.get("generation_count") or 0) >= 5
            and max(mainline_cases or [0]) >= 100,
            score=_mean([
                _cap((recursive.get("generation_count") or 0) / 5),
                _cap(max(mainline_cases or [0]) / 100),
                float(bool(recursive.get("pass"))),
            ]),
            evidence={
                "generation_count": recursive.get("generation_count"),
                "branch_test_count": recursive.get("branch_test_count"),
                "max_mainline_case_count": max(mainline_cases or [0]),
                "best_base_delta": recursive.get("metrics", {}).get("best_base_delta"),
                "final_placebo_delta": recursive.get("metrics", {}).get("final_placebo_delta"),
            },
            remaining_gap="Run the same chain on a fresh unseen task suite, not only the current structural live sequence.",
        ),
        _gate(
            "multi_hypothesis_generation",
            int(surface.get("proposal_count") or 0) >= 7
            and int(trace_policy.get("proposal_count") or 0) >= 5
            and float(trajectory.get("multi_path_rate") or 0.0) >= 0.7
            and int(preflight.get("ready_count") or 0) >= 5,
            score=_mean([
                _cap((surface.get("proposal_count") or 0) / 7),
                _cap((trace_policy.get("proposal_count") or 0) / 5),
                _cap(float(trajectory.get("multi_path_rate") or 0.0) / 0.7),
                _cap((preflight.get("ready_count") or 0) / 5),
            ]),
            evidence={
                "surface_proposal_count": surface.get("proposal_count"),
                "trace_policy_proposal_count": trace_policy.get("proposal_count"),
                "trace_policy_ready_count": preflight.get("ready_count"),
                "multi_path_rate": trajectory.get("multi_path_rate"),
                "top_path_label_hit_rate": trajectory.get("top_path_label_hit_rate"),
            },
            remaining_gap="Add more non-template hypothesis families synthesized from evaluator/world-model residuals.",
        ),
        _gate(
            "novelty_and_integration_classification",
            bool(novelty.get("pass"))
            and int(novelty.get("proposal_count") or 0) >= 5
            and all(novelty_counts.get(name, 0) >= 1 for name in [
                "duplicate",
                "specialization",
                "formal_isomorphism",
                "analogy",
                "genuinely_new_family",
            ]),
            score=float(novelty.get("gold_accuracy") or 0.0),
            evidence={
                "gold_accuracy": novelty.get("gold_accuracy"),
                "classification_counts": novelty_counts,
                "recommended_edge_counts": novelty.get("recommended_edge_counts"),
            },
            remaining_gap="Validate classification on real proposal batches, not only the deterministic gold fixture.",
        ),
        _gate(
            "fresh_ablation_controls_and_v5",
            bool(verifier.get("pass"))
            and int(verifier.get("accepted_count") or 0) >= 2
            and int(verifier.get("rejected_count") or 0) >= 10
            and bool(verifier.get("accepted_protocol_ok"))
            and bool(verifier.get("rejected_protocol_ok"))
            and bool(verifier.get("external_objective_gate_ok")),
            score=_mean([
                _cap((verifier.get("accepted_count") or 0) / 2),
                _cap((verifier.get("rejected_count") or 0) / 10),
                float(bool(verifier.get("accepted_protocol_ok"))),
                float(bool(verifier.get("rejected_protocol_ok"))),
                float(bool(verifier.get("external_objective_gate_ok"))),
            ]),
            evidence={
                "proposal_count": verifier.get("proposal_count"),
                "accepted_count": verifier.get("accepted_count"),
                "rejected_count": verifier.get("rejected_count"),
                "falsification_experiment_count": verifier.get("falsification_experiment_count"),
                "external_objective_gate_ok": verifier.get("external_objective_gate_ok"),
            },
            remaining_gap="Replace deterministic external-V5 controls with live heldout objective tasks.",
        ),
        _gate(
            "recursive_readback_and_gated_retention",
            bool(daemon.get("pass"))
            and bool(daemon.get("real_artifact_readback_resumed"))
            and int(daemon.get("real_artifact_readback_trigger_judgment_count") or 0) >= 3
            and int(daemon.get("real_artifact_readback_control_loss_count") or 0) == 0
            and int(daemon.get("accepted_apply_count") or 0) >= 2,
            score=_mean([
                float(bool(daemon.get("pass"))),
                float(bool(daemon.get("real_artifact_readback_resumed"))),
                _cap((daemon.get("real_artifact_readback_trigger_judgment_count") or 0) / 3),
                float(int(daemon.get("real_artifact_readback_control_loss_count") or 0) == 0),
                _cap((daemon.get("accepted_apply_count") or 0) / 2),
            ]),
            evidence={
                "accepted_apply_count": daemon.get("accepted_apply_count"),
                "real_artifact_readback_accept_count": daemon.get("real_artifact_readback_accept_count"),
                "real_artifact_readback_trigger_judgment_count": daemon.get("real_artifact_readback_trigger_judgment_count"),
                "real_artifact_readback_control_judgment_count": daemon.get("real_artifact_readback_control_judgment_count"),
                "real_artifact_readback_control_loss_count": daemon.get("real_artifact_readback_control_loss_count"),
            },
            remaining_gap="Scale daemon readback to repeated unattended batches under explicit budget policies.",
        ),
        _gate(
            "next_generation_productivity",
            bool(recursive.get("pass"))
            and int(recursive.get("accepted_mainline_count") or 0) >= 4
            and int(recursive.get("rejected_branch_count") or 0) >= 1
            and float(recursive.get("metrics", {}).get("best_base_delta") or 0.0) >= 0.10,
            score=_mean([
                _cap((recursive.get("accepted_mainline_count") or 0) / 4),
                _cap((recursive.get("rejected_branch_count") or 0) / 1),
                _cap(float(recursive.get("metrics", {}).get("best_base_delta") or 0.0) / 0.10),
            ]),
            evidence={
                "accepted_mainline_count": recursive.get("accepted_mainline_count"),
                "rejected_branch_count": recursive.get("rejected_branch_count"),
                "mainline_best_trace": recursive.get("metrics", {}).get("mainline_best_trace"),
            },
            remaining_gap="Demonstrate this over multiple independent clades, not just the structural repair sequence.",
        ),
        _gate(
            "morphism_independent_contribution",
            bool(morphism.get("pass"))
            and float(morphism.get("morphism_margin_over_best_baseline") or 0.0) >= 0.20,
            score=_mean([
                float(bool(morphism.get("pass"))),
                _cap((morphism.get("morphism_margin_over_best_baseline") or 0.0) / 0.20),
                _cap((morphism.get("nonlexical_structural_success_rate") or 0.0) / 0.75),
            ]),
            evidence={
                "scorer_hit_rates": morphism.get("scorer_hit_rates"),
                "morphism_margin_over_best_baseline": morphism.get("morphism_margin_over_best_baseline"),
                "nonlexical_structural_success_rate": morphism.get("nonlexical_structural_success_rate"),
            },
            remaining_gap="Show downstream performance gain from morphism retrieval on live tasks, beyond retrieval hit rate.",
        ),
        _gate(
            "cheap_world_model_calibration",
            bool(world.get("pass"))
            and (world.get("pre_acceptance") or {}).get("auc") is not None
            and float((world.get("pre_acceptance") or {}).get("auc") or 0.0) >= 0.85
            and float((world.get("post_calibration") or {}).get("brier_score") or 1.0) <= 0.08
            and float(trace_outcome.get("best_brier_score") or 1.0) <= 0.12,
            score=_mean([
                _cap(float((world.get("pre_acceptance") or {}).get("auc") or 0.0) / 0.85),
                _cap(0.08 / max(float((world.get("post_calibration") or {}).get("brier_score") or 1.0), 0.0001)),
                _cap(0.12 / max(float(trace_outcome.get("best_brier_score") or 1.0), 0.0001)),
            ]),
            evidence={
                "world_model_labeled_count": world.get("matched_label_count"),
                "world_model_auc": (world.get("pre_acceptance") or {}).get("auc"),
                "world_model_brier": (world.get("post_calibration") or {}).get("brier_score"),
                "trace_outcome_best_brier": trace_outcome.get("best_brier_score"),
                "trace_policy_update_count": trace_outcome.get("policy_update_count"),
            },
            remaining_gap="The calibrated rows still depend heavily on distilled transitions; raw first-party coverage is scored separately.",
        ),
    ]


def _research_gap_gates(
    *,
    sections: dict[str, dict],
    recursive: dict,
    morphism: dict,
    novelty: dict,
    residual_calibration: dict,
    formal_depth: dict,
) -> list[dict]:
    trace_dataset = sections.get("trace_dataset", {})
    trace_outcome = sections.get("trace_outcome_model", {})
    surface = sections.get("surface_hypothesis_generator", {})
    residual = sections.get("residual_clusterer", {})
    daemon = sections.get("recursive_daemon", {})
    trajectory = sections.get("trajectory_search", {})
    raw_first_party = int(trace_dataset.get("raw_first_party_trainable_row_count") or 0)
    weighted_trainable = float(trace_dataset.get("weighted_trainable_row_count") or 0.0)
    return [
        _gate(
            "world_model_raw_first_party_scale",
            raw_first_party >= 1000 and float(trace_outcome.get("best_brier_score") or 1.0) <= 0.12,
            score=_mean([
                _cap(raw_first_party / 1000),
                _cap(weighted_trainable / 1000),
                _cap(0.12 / max(float(trace_outcome.get("best_brier_score") or 1.0), 0.0001)),
            ]),
            evidence={
                "raw_first_party_trainable_row_count": raw_first_party,
                "first_party_distilled_trainable_row_count": trace_dataset.get("first_party_distilled_trainable_row_count"),
                "artifact_replay_trainable_row_count": trace_dataset.get("artifact_replay_trainable_row_count"),
                "weighted_trainable_row_count": trace_dataset.get("weighted_trainable_row_count"),
                "best_brier_score": trace_outcome.get("best_brier_score"),
            },
            remaining_gap="Collect about 1000 independent raw first-party recursive/daemon traces; current 1000+ rows are mostly distilled.",
        ),
        _gate(
            "creative_hypothesis_generator_loop",
            int(surface.get("synthesis_family_count") or 0) >= 4
            and int(surface.get("proposal_count") or 0) >= 7
            and novelty.get("classification_counts", {}).get("genuinely_new_family", 0) >= 1
            and float(trajectory.get("top_path_label_hit_rate") or 0.0) >= 0.85,
            score=_mean([
                _cap((surface.get("synthesis_family_count") or 0) / 4),
                _cap((surface.get("proposal_count") or 0) / 7),
                _cap(novelty.get("classification_counts", {}).get("genuinely_new_family", 0)),
                _cap(float(trajectory.get("top_path_label_hit_rate") or 0.0) / 0.85),
            ]),
            evidence={
                "synthesis_family_count": surface.get("synthesis_family_count"),
                "proposal_count": surface.get("proposal_count"),
                "new_family_fixture_count": novelty.get("classification_counts", {}).get("genuinely_new_family"),
                "top_path_label_hit_rate": trajectory.get("top_path_label_hit_rate"),
            },
            remaining_gap="Use evaluator/world-model residual clusters to create novel method families across multiple real generations.",
        ),
        _gate(
            "continuous_daemon_autonomy",
            False,
            score=_mean([
                float(bool(daemon.get("preflight_queue_consumed"))),
                _cap((daemon.get("preflight_queue_ready_count") or 0) / 5),
                float(bool(daemon.get("bounded_execute_resumed"))),
                0.0,
            ]),
            evidence={
                "preflight_queue_consumed": daemon.get("preflight_queue_consumed"),
                "preflight_queue_ready_count": daemon.get("preflight_queue_ready_count"),
                "bounded_execute_resumed": daemon.get("bounded_execute_resumed"),
                "continuous_background_mode": False,
            },
            remaining_gap="Implement a budgeted continuous daemon that repeatedly executes ready proposals, ingests judgments, clusters residuals, and queues next proposals.",
        ),
        _gate(
            "residual_label_large_scale_calibration",
            bool(residual_calibration.get("pass"))
            and int(residual_calibration.get("example_count") or 0) >= 100
            and float(residual_calibration.get("macro_f1") or 0.0) >= 0.85,
            score=_mean([
                _cap((residual_calibration.get("example_count") or 0) / 100),
                _cap(float(residual_calibration.get("macro_f1") or 0.0) / 0.85),
                _cap((residual.get("record_count") or 0) / 100),
            ]),
            evidence={
                "large_calibration_pass": residual_calibration.get("pass"),
                "large_calibration_example_count": residual_calibration.get("example_count"),
                "large_calibration_macro_f1": residual_calibration.get("macro_f1"),
                "large_calibration_accuracy": residual_calibration.get("accuracy"),
                "large_calibration_label_source_counts": residual_calibration.get("label_source_counts"),
                "legacy_label_agreement_example_count": residual.get("label_agreement_example_count"),
                "legacy_label_agreement_macro_f1": residual.get("label_agreement_macro_f1"),
                "record_count": residual.get("record_count"),
                "cluster_count": residual.get("cluster_count"),
            },
            remaining_gap="Future work should replace graph/trace-derived labels with a larger human/LLM-adjudicated residual set.",
        ),
        _gate(
            "formal_engine_depth",
            bool(formal_depth.get("pass")),
            score=_mean([
                float(bool(formal_depth.get("pass"))),
                _cap((formal_depth.get("summary", {}).get("complete_mapping_count") or 0) / 5),
                _cap((formal_depth.get("summary", {}).get("negative_control_application_count") or 0) / 200),
                _cap(float(formal_depth.get("summary", {}).get("downstream_transfer_auc") or 0.0) / 0.90),
                _cap((morphism.get("morphism_margin_over_best_baseline") or 0.0) / 0.20),
            ]),
            evidence={
                "bounded_formal_engine_depth_pass": formal_depth.get("bounded_formal_engine_depth_pass"),
                "formal_depth_summary": formal_depth.get("summary"),
                "formal_depth_gate_failures": [
                    gate.get("gate") for gate in formal_depth.get("gates", []) if not gate.get("pass")
                ],
                "bounded_morphism_benchmark_pass": morphism.get("pass"),
                "morphism_margin_over_best_baseline": morphism.get("morphism_margin_over_best_baseline"),
                "full_category_theory_solver": formal_depth.get("strict_category_theory_theorem_prover"),
                "true_blackwell_or_fisher_engine": formal_depth.get("true_blackwell_or_fisher_engine"),
            },
            remaining_gap="Bounded formal depth passes; future work can add strict theorem proving, exact Blackwell comparison, or richer information geometry.",
        ),
    ]


def _completion_estimates(*, line_gates: list[dict], gap_gates: list[dict]) -> dict:
    line_score = _mean(gate["score"] for gate in line_gates)
    gap_score = _mean(gate["score"] for gate in gap_gates)
    return {
        "recursive_hypothesis_argument_percent": round(100 * _bounded(0.70 + 0.22 * line_score), 1),
        "general_hypothesis_os_percent": round(100 * _bounded(0.40 + 0.35 * gap_score), 1),
        "reconstruction_md_behavior_percent": round(100 * _bounded(0.78 + 0.16 * line_score), 1),
        "interpretation": (
            "The recursive benchmark line is now well supported, but the general Assumption OS score remains lower "
            "because some paper-level gaps, such as raw first-party world-model data or continuous autonomy, are not complete."
        ),
    }


def _gate(name: str, passed: bool, *, score: float, evidence: dict, remaining_gap: str) -> dict:
    return {
        "name": name,
        "pass": bool(passed),
        "score": round(_bounded(score), 4),
        "evidence": evidence,
        "remaining_gap": remaining_gap,
    }


def _recursive_summary(payload: dict) -> dict:
    return {
        "pass": payload.get("pass"),
        "generation_count": payload.get("generation_count"),
        "branch_test_count": payload.get("branch_test_count"),
        "accepted_mainline_count": payload.get("accepted_mainline_count"),
        "rejected_branch_count": payload.get("rejected_branch_count"),
        "metrics": payload.get("metrics"),
    }


def _morphism_summary(payload: dict) -> dict:
    return {
        "pass": payload.get("pass"),
        "case_count": payload.get("case_count"),
        "scorer_hit_rates": payload.get("scorer_hit_rates"),
        "morphism_margin_over_best_baseline": payload.get("morphism_margin_over_best_baseline"),
        "nonlexical_structural_success_rate": payload.get("nonlexical_structural_success_rate"),
    }


def _novelty_summary(payload: dict) -> dict:
    return {
        "pass": payload.get("pass"),
        "proposal_count": payload.get("proposal_count"),
        "gold_accuracy": payload.get("gold_accuracy"),
        "classification_counts": payload.get("classification_counts"),
        "recommended_edge_counts": payload.get("recommended_edge_counts"),
    }


def _large_residual_summary(payload: dict) -> dict:
    return {
        "pass": payload.get("pass"),
        "example_count": payload.get("example_count"),
        "label_count": payload.get("label_count"),
        "accuracy": payload.get("accuracy"),
        "macro_f1": payload.get("macro_f1"),
        "label_source_counts": payload.get("label_source_counts"),
        "expected_type_counts": payload.get("expected_type_counts"),
    }


def _formal_depth_summary(payload: dict) -> dict:
    return {
        "pass": payload.get("pass"),
        "summary": payload.get("summary"),
        "gate_count": len(payload.get("gates", [])),
        "failed_gates": [gate.get("gate") for gate in payload.get("gates", []) if not gate.get("pass")],
        "strict_category_theory_theorem_prover": payload.get("strict_category_theory_theorem_prover"),
        "true_blackwell_or_fisher_engine": payload.get("true_blackwell_or_fisher_engine"),
    }


def _performance_summary(sections: dict[str, dict]) -> dict:
    names = [
        "world_model",
        "trace_dataset",
        "trace_outcome_model",
        "surface_hypothesis_generator",
        "trace_policy_proposals",
        "trace_policy_preflight",
        "recursive_daemon",
        "residual_clusterer",
        "verifier_stack",
        "assumption_bench",
    ]
    return {
        name: {
            key: value
            for key, value in (sections.get(name, {}) or {}).items()
            if key in {
                "pass",
                "proposal_count",
                "ready_count",
                "accepted_count",
                "rejected_count",
                "accepted_apply_count",
                "raw_first_party_trainable_row_count",
                "first_party_distilled_trainable_row_count",
                "weighted_trainable_row_count",
                "best_brier_score",
                "overall_score",
                "label_agreement_example_count",
                "label_agreement_macro_f1",
            }
        }
        for name in names
    }


def _next_actions(gap_gates: list[dict]) -> list[dict]:
    failed = [gate for gate in gap_gates if not gate["pass"]]
    failed.sort(key=lambda gate: (gate["score"], gate["name"]))
    actions = []
    for gate in failed:
        actions.append({
            "gap": gate["name"],
            "score": gate["score"],
            "next_action": gate["remaining_gap"],
        })
    return actions


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"performance payload not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _cap(value: float | int | None) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values) -> float:
    rows = [float(v) for v in values]
    return sum(rows) / len(rows) if rows else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the paper-facing recursive assumption benchmark line.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--performance-payload", default=str(DEFAULT_PERFORMANCE_PATH))
    ap.add_argument("--eval-id", default="paper_benchmark_line_20260604")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    graph_dir = (root / args.graph_dir).resolve() if not Path(args.graph_dir).is_absolute() else Path(args.graph_dir)
    perf_path = (root / args.performance_payload).resolve() if not Path(args.performance_payload).is_absolute() else Path(args.performance_payload)
    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    payload = build_paper_benchmark_line_payload(
        root=root,
        graph_dir=graph_dir,
        eval_id=args.eval_id,
        performance_path=perf_path,
    )
    _write_json(out_path, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "benchmark_line_pass": payload["benchmark_line_pass"],
        "research_gap_pass": payload["research_gap_pass"],
        "paper_readiness_pass": payload["paper_readiness_pass"],
        "completion_estimates": payload["completion_estimates"],
        "failed_research_gaps": [g["name"] for g in payload["research_gap_gates"] if not g["pass"]],
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
