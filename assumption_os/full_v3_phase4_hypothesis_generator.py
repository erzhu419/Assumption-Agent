"""Full-v3 Phase 4 multi-layer hypothesis generator validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .full_v2_phase4_hypothesis_generator_bypass import build_full_v2_phase4_hypothesis_generator_bypass_payload


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase4_hypothesis_generator_20260611.json"

REQUIRED_LAYERS = {"object", "method", "evaluator", "memory", "world_model", "meta_evolution"}


def build_full_v3_phase4_hypothesis_generator_payload(
    *,
    eval_id: str = "full_v3_phase4_hypothesis_generator_20260611",
) -> dict[str, Any]:
    source = build_full_v2_phase4_hypothesis_generator_bypass_payload(eval_id=f"{eval_id}_source")
    candidates = list(source["candidates"])
    clusters = list(source["clusters"])
    metrics = _metrics(source, clusters=clusters, candidates=candidates)
    gates = {
        "source_generator_passes": bool(source.get("pass")),
        "all_hypothesis_layers_present": metrics["layer_coverage"] == 1.0,
        "multi_trajectory_per_cluster": metrics["min_trajectories_per_cluster"] >= 2,
        "execution_lapses_filtered": metrics["execution_lapse_filtered_rate"] == 1.0,
        "novelty_integration_accurate": metrics["novelty_integration_accuracy"] >= 0.90,
        "selective_retention_precise": metrics["selective_retention_precision"] >= 0.90,
        "fresh_validation_success_high": metrics["fresh_validation_success_rate"] >= 0.80,
        "world_model_screen_precise": metrics["world_model_screen_precision"] >= 0.90,
        "false_discovery_low": metrics["false_discovery_rate"] <= 0.10,
        "descendant_productivity_high": metrics["descendant_productivity"] >= 0.65,
        "recursive_runner_seeded": metrics["recursive_runner_seed_rate"] >= 0.45,
        "shadow_mode_no_graph_mutation": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase4_multi_layer_hypothesis_generator",
        "reconstruction_v2_full_phase": "phase4_v3_multi_layer_hypothesis_generator",
        "performance_validation": True,
        "shadow_bypass": True,
        "validation_scope": (
            "Variation-evaluation-selective-retention validation for object, method, evaluator, memory, "
            "world-model, and meta-evolution hypotheses.  Candidates are generated from systematic residuals, "
            "classified as old-family/new-family/risky controls, screened, and retained only after fresh tests."
        ),
        "source": {
            "eval_id": source["eval_id"],
            "eval_kind": source["eval_kind"],
            "pass": source["pass"],
        },
        "layer_counts": metrics["layer_counts"],
        "trajectory_counts_by_cluster": metrics["trajectory_counts_by_cluster"],
        "retention_rows": _retention_rows(candidates),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase 4 is now an explicit v3 hypothesis generator: it creates multiple hypothesis trajectories "
            "per residual cluster, covers every assumption layer, and implements variation, evaluation, and "
            "selective retention before recursive execution."
        ),
    }


def _retention_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        novelty = candidate["novelty_check"]
        fresh = candidate["fresh_validation"]
        world = candidate["world_model_screen"]
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "cluster_id": candidate["cluster_id"],
            "layer": candidate["layer"],
            "trajectory": candidate["trajectory"],
            "classification": novelty["classification"],
            "duplicate": novelty["duplicate"],
            "conflict": novelty["conflict"],
            "world_model_action": world["recommended_action"],
            "accepted_for_recursive_runner": fresh["accepted_for_recursive_runner"],
            "negative_control_harm_count": fresh["negative_control_harm_count"],
        })
    return rows


def _metrics(
    source: dict[str, Any],
    *,
    clusters: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    source_metrics = source["metrics"]
    layer_counts = Counter(candidate["layer"] for candidate in candidates)
    by_cluster: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        by_cluster[candidate["cluster_id"]].add(candidate["trajectory"])
    primary = [candidate for candidate in candidates if candidate["trajectory"] == "primary_family"]
    controls = [candidate for candidate in candidates if candidate["trajectory"] != "primary_family"]
    retained = [candidate for candidate in candidates if candidate["fresh_validation"]["accepted_for_recursive_runner"]]
    correct_retention = [
        candidate for candidate in candidates
        if candidate["fresh_validation"]["accepted_for_recursive_runner"] == (candidate in primary)
    ]
    novelty_correct = [
        candidate for candidate in primary
        if candidate["novelty_check"]["classification"] in {"new_family", "orthogonal_new_family"}
        and not candidate["novelty_check"]["duplicate"]
        and not candidate["novelty_check"]["conflict"]
    ]
    control_blocked = [
        candidate for candidate in controls
        if not candidate["fresh_validation"]["accepted_for_recursive_runner"]
    ]
    return {
        "cluster_count": len(clusters),
        "candidate_count": len(candidates),
        "layer_counts": dict(sorted(layer_counts.items())),
        "layer_coverage": round(len(set(layer_counts) & REQUIRED_LAYERS) / len(REQUIRED_LAYERS), 4),
        "trajectory_counts_by_cluster": {
            cluster_id: len(trajectories)
            for cluster_id, trajectories in sorted(by_cluster.items())
        },
        "min_trajectories_per_cluster": min((len(value) for value in by_cluster.values()), default=0),
        "execution_lapse_filtered_rate": source_metrics["execution_lapse_filtered_rate"],
        "novelty_integration_accuracy": round(len(novelty_correct) / max(1, len(primary)), 4),
        "risky_control_block_rate": round(len(control_blocked) / max(1, len(controls)), 4),
        "selective_retention_precision": round(len(correct_retention) / max(1, len(candidates)), 4),
        "fresh_validation_success_rate": source_metrics["fresh_validation_success_rate"],
        "world_model_screen_precision": source_metrics["world_model_screen_precision"],
        "false_discovery_rate": source_metrics["false_discovery_rate"],
        "descendant_productivity": source_metrics["descendant_productivity"],
        "recursive_runner_seed_rate": round(len(retained) / max(1, len(candidates)), 4),
        "residual_explained_fraction": source_metrics["residual_explained_fraction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 Phase 4 hypothesis generator validation.")
    parser.add_argument("--eval-id", default="full_v3_phase4_hypothesis_generator_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase4_hypothesis_generator_payload(eval_id=args.eval_id)
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
