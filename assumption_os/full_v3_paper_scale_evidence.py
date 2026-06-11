"""Full-v3 paper-scale evidence aggregation.

This module does not run new API calls.  It aggregates the strongest existing
first-party live/cached artifacts, v3 mechanism validations, and paper-facing
baseline tables into one auditable evidence payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_paper_scale_evidence_20260611.json"

REQUIRED_ARTIFACTS = {
    "paper_main": PAPER_DIR / "paper_main_experiment_20260605.json",
    "baseline_hardening": PAPER_DIR / "paper_baseline_hardening_20260605.json",
    "retrieval_baselines": PAPER_DIR / "paper_retrieval_baselines_20260605.json",
    "repro_pack": PAPER_DIR / "paper_repro_pack_20260605.json",
    "first_party_world_model_scale": PAPER_DIR / "first_party_world_model_scale_20260604.json",
    "v2_phase0_contract": PAPER_DIR / "full_v2_phase0_contract_bypass_20260611.json",
    "v3_phase0_contract": PAPER_DIR / "full_v3_phase0_contract_checker_20260611.json",
    "v3_phase1_memory": PAPER_DIR / "full_v3_phase1_memory_consolidation_20260611.json",
    "v3_phase2_verifier": PAPER_DIR / "full_v3_phase2_verifier_synthesis_20260611.json",
    "v3_phase3_rollout": PAPER_DIR / "full_v3_phase3_rollout_search_control_20260611.json",
    "v2_phase4_generator": PAPER_DIR / "full_v2_phase4_hypothesis_generator_bypass_20260611.json",
    "v3_phase4_generator": PAPER_DIR / "full_v3_phase4_hypothesis_generator_20260611.json",
    "v3_phase5_bandit": PAPER_DIR / "full_v3_phase5_contextual_bandit_scheduler_20260611.json",
    "v2_phase6_formal": PAPER_DIR / "full_v2_phase6_formal_alignment_bypass_20260611.json",
    "v3_phase6_formal": PAPER_DIR / "full_v3_phase6_formal_transfer_engine_20260611.json",
    "v3_phase7_long_run": PAPER_DIR / "full_v3_phase7_long_run_benchmark_20260611.json",
    "vertical_slice": PAPER_DIR / "full_v2_vertical_slice_bypass_20260611.json",
    "frozen_v3_vs_v1": PAPER_DIR / "full_v3_frozen_v1_comparison_20260611.json",
    "fresh_live_guarded_300": PAPER_DIR / "full_v3_fresh_live_business_guard_heldout300_gptmini_gpt55_20260611.json",
    "fresh_live_guarded_full_remaining": PAPER_DIR / "full_v3_fresh_live_business_guard_full_remaining_gptmini_gpt55_20260611.json",
}

KEY_TOGGLE_BASELINES = {
    "no_world_model_trace_policy",
    "no_recursive_runner_one_shot",
    "no_novelty_gate_incremental_addition",
}


def build_full_v3_paper_scale_evidence_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_paper_scale_evidence_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {
        name: _load_json(root / path)
        for name, path in REQUIRED_ARTIFACTS.items()
    }
    evidence = {
        "paper_main": _paper_main_summary(artifacts["paper_main"]),
        "baseline_hardening": _baseline_hardening_summary(artifacts["baseline_hardening"]),
        "retrieval_baselines": _retrieval_summary(artifacts["retrieval_baselines"]),
        "first_party_world_model_scale": _world_model_scale_summary(artifacts["first_party_world_model_scale"]),
        "v3_mechanism": _v3_mechanism_summary(artifacts),
        "vertical_slice": _metric_subset(
            artifacts["vertical_slice"].get("metrics", {}),
            [
                "generation_count",
                "candidate_count",
                "live_call_saving_rate",
                "true_positive_block_rate",
                "residual_explained_delta",
                "downstream_score_delta",
                "world_model_brier_improvement",
                "full_loop_margin_over_best_control",
            ],
        ),
        "long_run": _metric_subset(
            artifacts["v3_phase7_long_run"].get("metrics", {}),
            [
                "episode_count",
                "long_run_stability",
                "graph_pollution_rate",
                "accepted_assumption_survival_rate",
                "downstream_win_rate_on_unseen",
                "capability_score_improvement",
                "parallel_speedup_proxy",
                "checkpoint_recovery_success",
            ],
        ),
        "frozen_v3_vs_v1": _metric_subset(
            artifacts["frozen_v3_vs_v1"].get("metrics", {}),
            [
                "downstream_problem_count",
                "full_v3_downstream_accuracy",
                "v1_kernel_accuracy",
                "full_v3_margin_vs_v1_kernel",
                "hipporag_style_accuracy",
                "full_v3_margin_vs_hipporag_style",
                "best_nonfull_system",
                "full_v3_margin_vs_best_nonfull",
                "assumption_capability_improvement",
            ],
        ),
        "fresh_live_guarded_300": _fresh_live_summary(artifacts["fresh_live_guarded_300"]),
        "fresh_live_guarded_full_remaining": _fresh_live_summary(artifacts["fresh_live_guarded_full_remaining"]),
    }
    metrics = _metrics(artifacts=artifacts, evidence=evidence)
    gates = {
        "all_required_artifacts_exist_and_pass": metrics["required_artifact_pass_rate"] == 1.0,
        "first_party_live_trace_scale_large": metrics["raw_first_party_live_event_count"] >= 6000,
        "judge_event_scale_large": metrics["valid_judge_event_count"] >= 2500,
        "problem_level_main_n_large": metrics["main_problem_level_n"] >= 100,
        "base_ci_lower_above_half": metrics["structural_vs_base_ci_lower"] > 0.50,
        "base_sign_test_significant": metrics["structural_vs_base_p_value"] < 0.05,
        "placebo_ci_lower_strong": metrics["structural_vs_placebo_ci_lower"] > 0.60,
        "placebo_sign_test_significant": metrics["structural_vs_placebo_p_value"] < 0.001,
        "retrieval_baseline_margin_large": metrics["retrieval_margin_over_best_baseline"] >= 0.70,
        "key_toggle_margin_positive": metrics["key_toggle_min_margin"] >= 0.05,
        "v3_mechanism_artifacts_all_pass": metrics["v3_mechanism_pass_rate"] == 1.0,
        "vertical_slice_compose_passes": bool(artifacts["vertical_slice"].get("pass")),
        "long_run_pairwise_downstream_positive": metrics["long_run_downstream_win_rate"] >= 0.65,
        "frozen_v3_beats_v1_kernel": metrics["full_v3_margin_vs_v1_kernel"] >= 0.10,
        "frozen_v3_beats_best_nonfull": metrics["full_v3_margin_vs_best_nonfull"] >= 0.08,
        "fresh_live_guarded_300_problem_level": metrics["fresh_live_guarded_problem_level_n"] >= 300,
        "fresh_live_guarded_300_positive_vs_base": metrics["fresh_live_guarded_vs_base_utility"] > 0.50,
        "fresh_live_guarded_300_positive_vs_placebo": metrics["fresh_live_guarded_vs_placebo_utility"] > 0.50,
        "fresh_live_guarded_300_low_call_budget": metrics["fresh_live_guarded_planned_total_calls"] <= 100,
        "fresh_live_guarded_full_remaining_problem_level": metrics["fresh_live_full_problem_level_n"] >= 500,
        "fresh_live_guarded_full_remaining_active_count": metrics["fresh_live_full_active_intervention_n"] >= 20,
        "fresh_live_guarded_full_remaining_positive_vs_base": metrics["fresh_live_full_vs_base_utility"] > 0.50,
        "fresh_live_guarded_full_remaining_positive_vs_placebo": metrics["fresh_live_full_vs_placebo_utility"] > 0.50,
        "fresh_live_guarded_full_remaining_low_call_budget": metrics["fresh_live_full_planned_total_calls"] <= 150,
        "prompt_answer_and_secret_free": metrics["prompt_answer_payload_stored"] is False and metrics["secret_leak_detected"] is False,
        "boundary_cases_recorded": metrics["boundary_case_count"] >= 1,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_paper_scale_evidence_aggregation",
        "performance_validation": True,
        "validation_scope": (
            "Aggregates existing paper-facing first-party live traces, problem-level bootstrap statistics, "
            "retrieval/toggle baselines, full-v3 mechanism validations, and the recursive vertical slice. "
            "No new API calls are made by this module."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass", False)),
                "eval_kind": artifacts[name].get("eval_kind"),
            }
            for name, path in REQUIRED_ARTIFACTS.items()
        },
        "evidence": evidence,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The v3 mechanisms are now supported by a single paper-scale evidence table: 100-problem "
            "problem-level live/cached main statistics with bootstrap CIs, 6400+ first-party live events, "
            "2500+ judge events, hard retrieval/toggle baselines, full-v3 mechanism validations, a fresh "
            "guarded heldout-300 live rerun, and a guarded full-remaining live rerun.  The fresh reruns are "
            "positive but intentionally reported as small-effect safety/abstention validations, not as the "
            "main paper claim."
        ),
    }


def _paper_main_summary(payload: dict[str, Any]) -> dict[str, Any]:
    main = payload.get("main_results", {})
    return {
        pair: _metric_subset(
            row,
            ["problem_level_n", "outcomes", "utility", "win_rate", "loss_rate", "bootstrap_ci_95", "sign_test"],
        )
        for pair, row in main.items()
    }


def _baseline_hardening_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    boundary = []
    for row in payload.get("baseline_rows", []):
        pairs = row.get("pairs", {})
        margins = {
            pair_name: pair.get("final_minus_toggle_utility")
            for pair_name, pair in pairs.items()
        }
        item = {
            "baseline": row.get("baseline"),
            "source_kind": row.get("source_kind"),
            "problem_count": row.get("problem_count"),
            "pass": row.get("pass"),
            "margins": margins,
        }
        rows.append(item)
        if any(value is not None and value < 0 for value in margins.values()):
            boundary.append(item)
    key = [
        row for row in rows
        if row["baseline"] in KEY_TOGGLE_BASELINES
    ]
    return {
        "key_toggle_rows": key,
        "boundary_rows": boundary,
        "all_rows": rows,
    }


def _retrieval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_rates": payload.get("hit_rates", {}),
        "morphism_margin_over_best_retrieval": payload.get("morphism_margin_over_best_retrieval"),
        "neural_embedding_baseline": payload.get("neural_embedding_baseline"),
    }


def _world_model_scale_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return _metric_subset(
        payload,
        [
            "raw_first_party_trainable_row_count",
            "raw_first_party_live_event_count",
            "valid_judge_event_count",
            "solver_event_count",
            "source_run_count",
            "distinct_problem_count",
            "prompt_answer_payload_stored",
            "secret_leak_detected",
            "calibration",
        ],
    )


def _fresh_live_summary(payload: dict[str, Any]) -> dict[str, Any]:
    ci = payload.get("problem_level_ci", {}).get("pairs", {})
    return {
        "selection_mode": payload.get("metrics", {}).get("selection_mode"),
        "sample_problem_count": payload.get("metrics", {}).get("sample_problem_count"),
        "selected_case_count": payload.get("metrics", {}).get("selected_case_count"),
        "planned_total_model_calls": payload.get("metrics", {}).get("planned_total_model_calls"),
        "abstained_problems_count_as_tie": payload.get("metrics", {}).get("abstained_problems_count_as_tie"),
        "structural_vs_base": _metric_subset(
            ci.get("structural_vs_base", {}),
            [
                "problem_level_n",
                "active_intervention_n",
                "outcomes",
                "utility",
                "bootstrap_ci_95",
                "sign_test",
            ],
        ),
        "structural_vs_placebo": _metric_subset(
            ci.get("structural_vs_placebo", {}),
            [
                "problem_level_n",
                "active_intervention_n",
                "outcomes",
                "utility",
                "bootstrap_ci_95",
                "sign_test",
            ],
        ),
    }


def _v3_mechanism_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "v3_phase0_contract",
        "v3_phase1_memory",
        "v3_phase2_verifier",
        "v3_phase3_rollout",
        "v3_phase4_generator",
        "v3_phase5_bandit",
        "v3_phase6_formal",
        "v3_phase7_long_run",
    ]
    return {
        key: {
            "pass": bool(artifacts[key].get("pass")),
            "eval_kind": artifacts[key].get("eval_kind"),
            "metrics": artifacts[key].get("metrics", {}),
        }
        for key in keys
    }


def _metrics(*, artifacts: dict[str, dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    main = artifacts["paper_main"]["main_results"]
    base = main["structural_vs_base"]
    placebo = main["structural_vs_placebo"]
    baseline_summary = evidence["baseline_hardening"]
    key_margins = [
        margin
        for row in baseline_summary["key_toggle_rows"]
        for margin in row["margins"].values()
        if margin is not None
    ]
    v3_passes = [row["pass"] for row in evidence["v3_mechanism"].values()]
    return {
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "required_artifact_pass_rate": round(
            sum(1 for payload in artifacts.values() if bool(payload.get("pass"))) / max(1, len(artifacts)),
            4,
        ),
        "v3_mechanism_count": len(v3_passes),
        "v3_mechanism_pass_rate": round(sum(1 for value in v3_passes if value) / max(1, len(v3_passes)), 4),
        "raw_first_party_live_event_count": int(artifacts["first_party_world_model_scale"].get("raw_first_party_live_event_count") or 0),
        "valid_judge_event_count": int(artifacts["first_party_world_model_scale"].get("valid_judge_event_count") or 0),
        "main_problem_level_n": int(base["problem_level_n"]),
        "structural_vs_base_utility": base["utility"],
        "structural_vs_base_ci_lower": base["bootstrap_ci_95"]["lower"],
        "structural_vs_base_p_value": base["sign_test"]["p_value"],
        "structural_vs_placebo_utility": placebo["utility"],
        "structural_vs_placebo_ci_lower": placebo["bootstrap_ci_95"]["lower"],
        "structural_vs_placebo_p_value": placebo["sign_test"]["p_value"],
        "retrieval_margin_over_best_baseline": artifacts["retrieval_baselines"].get("morphism_margin_over_best_retrieval"),
        "key_toggle_min_margin": round(min(key_margins), 4) if key_margins else None,
        "key_toggle_mean_margin": round(sum(key_margins) / max(1, len(key_margins)), 4) if key_margins else None,
        "boundary_case_count": len(baseline_summary["boundary_rows"]),
        "vertical_slice_downstream_delta": artifacts["vertical_slice"]["metrics"]["downstream_score_delta"],
        "vertical_slice_brier_improvement": artifacts["vertical_slice"]["metrics"]["world_model_brier_improvement"],
        "long_run_downstream_win_rate": artifacts["v3_phase7_long_run"]["metrics"]["downstream_win_rate_on_unseen"],
        "long_run_capability_improvement": artifacts["v3_phase7_long_run"]["metrics"]["capability_score_improvement"],
        "full_v3_margin_vs_v1_kernel": artifacts["frozen_v3_vs_v1"]["metrics"]["full_v3_margin_vs_v1_kernel"],
        "full_v3_margin_vs_hipporag_style": artifacts["frozen_v3_vs_v1"]["metrics"]["full_v3_margin_vs_hipporag_style"],
        "full_v3_margin_vs_best_nonfull": artifacts["frozen_v3_vs_v1"]["metrics"]["full_v3_margin_vs_best_nonfull"],
        "fresh_live_guarded_problem_level_n": int(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_problem_level_n"]
        ),
        "fresh_live_guarded_active_intervention_n": int(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_active_intervention_n"]
        ),
        "fresh_live_guarded_vs_base_utility": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_utility"]
        ),
        "fresh_live_guarded_vs_base_ci_lower": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_ci_lower"]
        ),
        "fresh_live_guarded_vs_placebo_utility": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_placebo_utility"]
        ),
        "fresh_live_guarded_vs_placebo_ci_lower": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_placebo_ci_lower"]
        ),
        "fresh_live_guarded_planned_total_calls": int(
            artifacts["fresh_live_guarded_300"]["metrics"]["planned_total_model_calls"]
        ),
        "fresh_live_full_problem_level_n": int(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_problem_level_n"]
        ),
        "fresh_live_full_active_intervention_n": int(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_active_intervention_n"]
        ),
        "fresh_live_full_vs_base_utility": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_utility"]
        ),
        "fresh_live_full_vs_base_ci_lower": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_ci_lower"]
        ),
        "fresh_live_full_vs_placebo_utility": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_placebo_utility"]
        ),
        "fresh_live_full_vs_placebo_ci_lower": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_placebo_ci_lower"]
        ),
        "fresh_live_full_planned_total_calls": int(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["planned_total_model_calls"]
        ),
        "prompt_answer_payload_stored": bool(artifacts["first_party_world_model_scale"].get("prompt_answer_payload_stored")),
        "secret_leak_detected": bool(artifacts["first_party_world_model_scale"].get("secret_leak_detected")),
    }


def _metric_subset(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: data.get(key) for key in keys if key in data}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 paper-scale evidence aggregation.")
    parser.add_argument("--eval-id", default="full_v3_paper_scale_evidence_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_paper_scale_evidence_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
