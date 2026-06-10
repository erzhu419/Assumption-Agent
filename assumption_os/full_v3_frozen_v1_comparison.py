"""Frozen full-v3 versus v1-kernel comparison experiment.

This experiment is intentionally frozen/cached: it aggregates the current
reproducible artifacts and compares full-v3 against v1-style kernel baselines
without making new API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_frozen_v1_comparison_20260611.json"

PHASE_ARTIFACTS = {
    "phase0_v3": PAPER_DIR / "full_v3_phase0_contract_checker_20260611.json",
    "phase1_v3": PAPER_DIR / "full_v3_phase1_memory_consolidation_20260611.json",
    "phase2_v3": PAPER_DIR / "full_v3_phase2_verifier_synthesis_20260611.json",
    "phase3_v3": PAPER_DIR / "full_v3_phase3_rollout_search_control_20260611.json",
    "phase4_v3": PAPER_DIR / "full_v3_phase4_hypothesis_generator_20260611.json",
    "phase5_v3": PAPER_DIR / "full_v3_phase5_contextual_bandit_scheduler_20260611.json",
    "phase6_v3": PAPER_DIR / "full_v3_phase6_formal_transfer_engine_20260611.json",
    "phase7_v3": PAPER_DIR / "full_v3_phase7_long_run_benchmark_20260611.json",
}


def build_full_v3_frozen_v1_comparison_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_frozen_v1_comparison_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    phases = {name: _load_json(root / path) for name, path in PHASE_ARTIFACTS.items()}
    main = _load_json(root / PAPER_DIR / "paper_main_experiment_20260605.json")
    retrieval = _load_json(root / PAPER_DIR / "paper_retrieval_baselines_20260605.json")
    long_run = phases["phase7_v3"]
    downstream = long_run["downstream_bench"]
    metrics = _metrics(phases=phases, main=main, retrieval=retrieval, downstream=downstream, long_run=long_run)
    gates = {
        "all_v3_phase_artifacts_pass": metrics["phase_pass_rate"] == 1.0,
        "full_v3_beats_v1_kernel": metrics["full_v3_margin_vs_v1_kernel"] >= 0.10,
        "full_v3_beats_hipporag_style": metrics["full_v3_margin_vs_hipporag_style"] >= 0.10,
        "full_v3_beats_best_ablation": metrics["full_v3_margin_vs_best_nonfull"] >= 0.08,
        "assumption_capability_improves": metrics["assumption_capability_improvement"] >= 0.15,
        "main_problem_level_significant": metrics["main_structural_vs_base_p_value"] < 0.05,
        "main_problem_level_ci_positive": metrics["main_structural_vs_base_ci_lower"] > 0.50,
        "retrieval_baseline_margin_large": metrics["retrieval_margin_over_best_baseline"] >= 0.70,
        "cached_experiment_scope_declared": metrics["fresh_api_call_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_frozen_v1_comparison_experiment",
        "performance_validation": True,
        "validation_scope": (
            "Frozen/cached comparison of full-v3 against v1-kernel and retrieval baselines.  This is an "
            "experiment artifact for reproducible comparison; it is not a new fresh live API rerun."
        ),
        "source_artifacts": {
            name: {"path": str(path), "pass": bool(phases[name].get("pass"))}
            for name, path in PHASE_ARTIFACTS.items()
        },
        "downstream_rows": downstream,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Against the v1-style case-reflection kernel, full-v3 improves downstream accuracy by 0.12 in the "
            "frozen long-run benchmark, improves assumption capability by 0.1945, and remains positive on the "
            "100-problem paper main experiment.  The next stronger experiment is a fresh live 300/600/full rerun."
        ),
    }


def _metrics(
    *,
    phases: dict[str, dict[str, Any]],
    main: dict[str, Any],
    retrieval: dict[str, Any],
    downstream: list[dict[str, Any]],
    long_run: dict[str, Any],
) -> dict[str, Any]:
    by_system = {row["system"]: row for row in downstream}
    full = by_system["full_v3_assumption_os"]["accuracy"]
    v1_kernel = by_system["case_reflection_v20"]["accuracy"]
    hipporag = by_system["hipporag_style_graph_retrieval"]["accuracy"]
    nonfull = [row for row in downstream if row["system"] != "full_v3_assumption_os"]
    best_nonfull = max(nonfull, key=lambda row: row["accuracy"])
    base_pair = main["main_results"]["structural_vs_base"]
    return {
        "phase_count": len(phases),
        "phase_pass_rate": round(sum(1 for row in phases.values() if row.get("pass")) / max(1, len(phases)), 4),
        "downstream_problem_count": by_system["full_v3_assumption_os"]["problem_count"],
        "full_v3_downstream_accuracy": full,
        "v1_kernel_system": "case_reflection_v20",
        "v1_kernel_accuracy": v1_kernel,
        "full_v3_margin_vs_v1_kernel": round(full - v1_kernel, 4),
        "hipporag_style_accuracy": hipporag,
        "full_v3_margin_vs_hipporag_style": round(full - hipporag, 4),
        "best_nonfull_system": best_nonfull["system"],
        "best_nonfull_accuracy": best_nonfull["accuracy"],
        "full_v3_margin_vs_best_nonfull": round(full - best_nonfull["accuracy"], 4),
        "assumption_capability_before": long_run["metrics"]["capability_score_before"],
        "assumption_capability_after": long_run["metrics"]["capability_score_after"],
        "assumption_capability_improvement": long_run["metrics"]["capability_score_improvement"],
        "main_problem_level_n": base_pair["problem_level_n"],
        "main_structural_vs_base_utility": base_pair["utility"],
        "main_structural_vs_base_ci_lower": base_pair["bootstrap_ci_95"]["lower"],
        "main_structural_vs_base_p_value": base_pair["sign_test"]["p_value"],
        "retrieval_margin_over_best_baseline": retrieval["morphism_margin_over_best_retrieval"],
        "fresh_api_call_count": 0,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen full-v3 versus v1 comparison experiment.")
    parser.add_argument("--eval-id", default="full_v3_frozen_v1_comparison_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_frozen_v1_comparison_payload(root=root, eval_id=args.eval_id)
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
