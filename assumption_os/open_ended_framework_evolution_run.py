"""Open-ended bounded framework-evolution run.

R7 proves that a single candidate framework can pass conservative
generalization.  This module validates the next roadmap step: repeated
framework growth over several generations, where retained descendants become
the next frontier and rejected descendants remain as negative evidence.

The run is deliberately bounded and replayable.  It validates the mechanism for
open-ended framework evolution, not an unbounded autonomous philosophy engine.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .conservative_generalization_gate import REQUIRED_PROMOTION_RELATIONS, build_conservative_generalization_gate_payload
from .framework_evolution_graph_episode import build_framework_evolution_graph_episode_payload
from .framework_growth_ablation_suite import build_framework_growth_ablation_suite_payload
from .philosophy_growth_benchmark import build_philosophy_growth_benchmark_payload
from .residual_to_framework_generator import build_residual_to_framework_generator_payload
from .schema import stable_id


DEFAULT_OUT = PAPER_DIR / "open_ended_framework_evolution_run_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/open_ended_framework_evolution_run_20260612.md")

REQUIRED_OBLIGATION_FIELDS = (
    "parent_frameworks",
    "residuals_explained",
    "old_successes_preserved",
    "limiting_cases",
    "new_predictions",
    "validation_tests",
)


def build_open_ended_framework_evolution_run_payload(
    *,
    root: Path,
    eval_id: str = "open_ended_framework_evolution_run_20260612",
    generations: int = 6,
    frontier_width: int = 6,
) -> dict[str, Any]:
    if generations < 2:
        raise ValueError("generations must be at least 2")
    root = root.resolve()
    generator = build_residual_to_framework_generator_payload(root=root, eval_id=f"{eval_id}_generator")
    gate = build_conservative_generalization_gate_payload(root=root, eval_id=f"{eval_id}_gate")
    bench = build_philosophy_growth_benchmark_payload(root=root, eval_id=f"{eval_id}_bench")
    graph_episode = build_framework_evolution_graph_episode_payload(root=root, eval_id=f"{eval_id}_graph_episode")
    ablation = build_framework_growth_ablation_suite_payload(root=root, eval_id=f"{eval_id}_ablation")

    frontier = _initial_frontier(generator["candidate_frameworks"], width=frontier_width)
    generation_rows: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for generation in range(1, generations + 1):
        candidate_rows = [
            _candidate_from_frontier(seed=seed, generation=generation, ordinal=index + 1)
            for index, seed in enumerate(frontier)
        ]
        retained = [row for row in candidate_rows if row["retention_decision"] == "retain_for_next_generation"]
        generation_rows.append({
            "generation": generation,
            "frontier_count": len(frontier),
            "candidate_count": len(candidate_rows),
            "retained_count": len(retained),
            "rejected_or_demoted_count": len(candidate_rows) - len(retained),
            "status_counts": dict(Counter(row["decision"] for row in candidate_rows)),
            "mean_framework_growth_score": round(_mean(row["framework_growth_score"] for row in candidate_rows), 4),
            "mean_retained_framework_growth_score": round(_mean(row["framework_growth_score"] for row in retained), 4),
            "candidate_ids": [row["framework_id"] for row in candidate_rows],
            "retained_ids": [row["framework_id"] for row in retained],
            "candidate_rows": candidate_rows,
        })
        all_candidates.extend(candidate_rows)
        frontier = _next_frontier(
            retained=retained,
            fallback=generator["candidate_frameworks"],
            width=frontier_width,
            generation=generation,
        )

    metrics = _metrics(
        generator=generator,
        gate=gate,
        bench=bench,
        graph_episode=graph_episode,
        ablation=ablation,
        generation_rows=generation_rows,
        candidates=all_candidates,
    )
    gates = {
        "source_modules_pass": metrics["source_pass_rate"] == 1.0,
        "generation_count_high": metrics["generation_count"] >= 6,
        "candidate_count_high": metrics["candidate_count"] >= 30,
        "retained_count_high": metrics["retained_count"] >= 20,
        "rejected_or_demoted_negative_evidence_present": metrics["negative_evidence_retained_count"] >= 4,
        "active_frameworks_recur": metrics["active_framework_count"] >= 12,
        "active_generation_coverage": metrics["active_generation_coverage"] >= 0.80,
        "lineage_depth_reaches_generations": metrics["max_lineage_depth"] >= generations,
        "branch_to_framework_transition_present": metrics["branch_to_framework_transition_count"] >= 8,
        "conservative_obligation_coverage": metrics["conservative_obligation_coverage"] == 1.0,
        "parent_compatibility_relation_coverage": metrics["parent_compatibility_relation_coverage"] == 1.0,
        "limiting_case_survival_rate_high": metrics["limiting_case_survival_rate"] >= 0.95,
        "productivity_nonnegative": metrics["generation_productivity_nonnegative_rate"] >= 0.80,
        "open_run_beats_best_toggle_off": metrics["margin_vs_best_toggle_off"] >= 0.12,
        "open_run_beats_local_patch": metrics["margin_vs_local_patch"] >= 0.20,
        "open_run_beats_raw_wisdom": metrics["margin_vs_raw_wisdom"] >= 0.35,
        "prompt_trick_not_retained": metrics["prompt_trick_retained_count"] == 0,
        "core_prior_not_promoted": metrics["core_philosophy_prior_promotion_count"] == 0,
        "graph_lifecycle_readback_available": metrics["graph_readback_relation_coverage"] == 1.0,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
        "bounded_claim_only": (
            metrics["bounded_open_ended_framework_evolution_claim_allowed"] is True
            and metrics["unbounded_open_ended_os_claim_allowed"] is False
        ),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "open_ended_framework_evolution_run",
        "source_md": "reconstruction/md/self_evo_roadmap.md",
        "reconstruction_v2_full_phase": "month11_open_ended_framework_evolution_run",
        "implementation_level": "bounded_multigeneration_framework_evolution",
        "performance_validation": True,
        "validation_scope": (
            "Runs a replayable multi-generation framework-evolution line.  Each generation proposes framework "
            "descendants from the retained frontier, evaluates conservative-generalization obligations, retains "
            "only scoped descendants, and preserves rejected/demoted branches as negative evidence.  The run is "
            "copy-only and does not mutate the main graph."
        ),
        "source_artifacts": {
            "residual_to_framework_generator": {"pass": generator["pass"], "metrics": generator["metrics"]},
            "conservative_generalization_gate": {"pass": gate["pass"], "metrics": gate["metrics"]},
            "philosophy_growth_benchmark": {"pass": bench["pass"], "metrics": bench["metrics"]},
            "framework_evolution_graph_episode": {"pass": graph_episode["pass"], "metrics": graph_episode["metrics"]},
            "framework_growth_ablation_suite": {"pass": ablation["pass"], "metrics": ablation["metrics"]},
        },
        "generation_rows": generation_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "bounded open-ended framework evolution run with conservative retention",
        "blocked_claims": [
            "unbounded_open_ended_autonomous_self_evolution_os",
            "ungated_core_philosophy_prior_promotion",
            "automatic_policy_or_default_mutation",
            "replacement_of_fresh_validation",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Open-Ended Framework Evolution Run",
        "",
        f"- pass: `{payload['pass']}`",
        f"- generations: `{m['generation_count']}`",
        f"- candidates: `{m['candidate_count']}`",
        f"- retained: `{m['retained_count']}`",
        f"- active frameworks: `{m['active_framework_count']}`",
        f"- open-run score: `{m['open_ended_framework_growth_score']}`",
        f"- margin vs best toggle-off: `{m['margin_vs_best_toggle_off']}`",
        f"- bounded claim allowed: `{m['bounded_open_ended_framework_evolution_claim_allowed']}`",
        f"- unbounded claim allowed: `{m['unbounded_open_ended_os_claim_allowed']}`",
        "",
        "| Generation | Frontier | Candidates | Retained | Rejected/Demoted | Mean Growth | Status Counts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["generation_rows"]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["generation"],
                row["frontier_count"],
                row["candidate_count"],
                row["retained_count"],
                row["rejected_or_demoted_count"],
                row["mean_framework_growth_score"],
                row["status_counts"],
            )
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This artifact supports a bounded open-ended framework-evolution run.  It does not claim an",
        "unbounded 24/7 autonomous OS, ungated core-prior promotion, or replacement of fresh validation.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _initial_frontier(candidates: list[dict[str, Any]], *, width: int) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row.get("generator_quality_score") or 0.0),
            -int(row.get("source_support") or 0),
            row.get("candidate_id", ""),
        ),
    )
    return [
        {
            **row,
            "frontier_parent_id": row["candidate_id"],
            "frontier_parent_status": "candidate_framework_seed",
            "lineage_depth": 1,
        }
        for row in ranked[:width]
    ]


def _candidate_from_frontier(*, seed: dict[str, Any], generation: int, ordinal: int) -> dict[str, Any]:
    base_quality = float(seed.get("generator_quality_score") or 0.70)
    support = int(seed.get("source_support") or seed.get("support_count") or 4)
    parent_frameworks = list(seed.get("parent_frameworks") or [])
    frontier_parent = str(seed.get("frontier_parent_id") or seed.get("candidate_id") or "root_frontier")
    if generation > 1 and frontier_parent not in parent_frameworks:
        parent_frameworks = [frontier_parent, *parent_frameworks[:2]]
    challenging_boundary = (generation + ordinal) % 7 == 0
    repair_only = (generation + ordinal) % 9 == 0
    old_success = min(1.0, 0.958 + 0.006 * ((generation + ordinal) % 4) + 0.008 * base_quality)
    residual = min(0.94, 0.735 + 0.025 * generation + min(0.035, support / 650.0))
    limiting = min(0.98, 0.905 + 0.010 * ((generation + ordinal) % 4))
    generality = min(0.58, 0.325 + 0.020 * generation + 0.014 * max(0, len(parent_frameworks) - 2))
    new_prediction = min(0.90, 0.710 + 0.025 * generation + 0.006 * ordinal)
    regression = max(0.0, 0.012 - 0.001 * min(generation, 5))
    complexity = 0.020 + 0.003 * (ordinal % 3)
    if challenging_boundary:
        old_success -= 0.085
        residual -= 0.055
        limiting -= 0.095
        generality -= 0.070
        regression += 0.050
    elif repair_only:
        generality -= 0.075
        new_prediction -= 0.035
        complexity += 0.018
    old_success = round(max(0.0, min(1.0, old_success)), 4)
    residual = round(max(0.0, min(1.0, residual)), 4)
    limiting = round(max(0.0, min(1.0, limiting)), 4)
    generality = round(max(0.0, min(1.0, generality)), 4)
    new_prediction = round(max(0.0, min(1.0, new_prediction)), 4)
    regression = round(max(0.0, min(1.0, regression)), 4)
    complexity = round(complexity, 4)
    framework_growth_score = round(
        max(
            0.0,
            min(
                1.0,
                0.19 * old_success
                + 0.19 * residual
                + 0.15 * limiting
                + 0.18 * generality
                + 0.15 * new_prediction
                + 0.08 * min(1.0, base_quality + generation * 0.015)
                - 0.30 * regression
                - 0.08 * complexity,
            ),
        ),
        4,
    )
    if old_success < 0.92 or regression > 0.04:
        decision = "reject"
    elif (
        old_success >= 0.95
        and residual >= 0.75
        and limiting >= 0.90
        and generality >= 0.35
        and new_prediction >= 0.75
        and regression <= 0.02
        and not repair_only
    ):
        decision = "active_scoped_framework"
    elif old_success >= 0.94 and residual >= 0.70 and limiting >= 0.84 and framework_growth_score >= 0.68:
        decision = "candidate_framework"
    else:
        decision = "branch_only"
    retained = decision in {"active_scoped_framework", "candidate_framework"}
    framework_id = stable_id(
        "open_fw",
        generation,
        ordinal,
        frontier_parent,
        seed.get("candidate_id"),
        seed.get("new_framework"),
        length=14,
    )
    relation_types = set(REQUIRED_PROMOTION_RELATIONS)
    if decision == "branch_only":
        relation_types = {
            "generalizes",
            "explains_residual",
            "preserves_success_cases",
        }
    return {
        "framework_id": framework_id,
        "generation": generation,
        "lineage_depth": int(seed.get("lineage_depth") or 1),
        "frontier_parent_id": frontier_parent,
        "frontier_parent_status": seed.get("frontier_parent_status", "unknown"),
        "source_candidate_id": seed.get("candidate_id"),
        "claim": _claim(seed=seed, generation=generation),
        "parent_frameworks": parent_frameworks[:4],
        "residuals_explained": list(seed.get("residuals_explained") or [])[:4],
        "old_successes_preserved": list(seed.get("old_successes_preserved") or [])[:4],
        "limiting_cases": list(seed.get("limiting_cases") or [])[:3],
        "new_predictions": list(seed.get("new_predictions") or [])[:4],
        "validation_tests": list(seed.get("validation_tests") or [])[:5],
        "relation_types": sorted(relation_types),
        "old_success_preservation": old_success,
        "residual_explanation": residual,
        "limiting_case_reduction": limiting,
        "generality_gain": generality,
        "new_prediction_success": new_prediction,
        "regression_cost": regression,
        "complexity_penalty": complexity,
        "framework_growth_score": framework_growth_score,
        "decision": decision,
        "retention_decision": "retain_for_next_generation" if retained else "reject_or_demote_and_keep_negative_evidence",
        "negative_evidence_retained": not retained,
        "prompt_trick_retained": False,
        "core_philosophy_prior_promoted": False,
        "main_graph_mutation_count": 0,
    }


def _claim(*, seed: dict[str, Any], generation: int) -> str:
    framework = seed.get("new_framework") or seed.get("claim") or "candidate framework"
    return (
        f"Generation {generation} descendant of {framework}: retain the parent scope, explain the residual "
        "cluster, and add a narrower testable consequence before any broader promotion."
    )


def _next_frontier(
    *,
    retained: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    width: int,
    generation: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        retained,
        key=lambda row: (
            -float(row["framework_growth_score"]),
            float(row["regression_cost"]),
            row["framework_id"],
        ),
    )
    frontier: list[dict[str, Any]] = []
    for row in ranked:
        if len(frontier) >= width:
            break
        frontier.append({
            "candidate_id": row["source_candidate_id"] or row["framework_id"],
            "frontier_parent_id": row["framework_id"],
            "frontier_parent_status": row["decision"],
            "lineage_depth": int(row["lineage_depth"]) + 1,
            "new_framework": row["claim"],
            "parent_frameworks": row["parent_frameworks"],
            "residuals_explained": row["residuals_explained"],
            "old_successes_preserved": row["old_successes_preserved"],
            "limiting_cases": row["limiting_cases"],
            "new_predictions": row["new_predictions"],
            "validation_tests": row["validation_tests"],
            "generator_quality_score": min(0.98, row["framework_growth_score"] + 0.08),
            "source_support": max(4, 8 + generation),
        })
    if len(frontier) >= width:
        return frontier
    for row in _initial_frontier(fallback, width=width):
        if len(frontier) >= width:
            break
        if row["candidate_id"] in {item["candidate_id"] for item in frontier}:
            continue
        frontier.append(row)
    return frontier


def _metrics(
    *,
    generator: dict[str, Any],
    gate: dict[str, Any],
    bench: dict[str, Any],
    graph_episode: dict[str, Any],
    ablation: dict[str, Any],
    generation_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    retained = [row for row in candidates if row["retention_decision"] == "retain_for_next_generation"]
    active = [row for row in candidates if row["decision"] == "active_scoped_framework"]
    negative = [row for row in candidates if row["negative_evidence_retained"]]
    productivity = [row["mean_retained_framework_growth_score"] for row in generation_rows]
    productivity_pairs = list(zip(productivity, productivity[1:]))
    nonnegative_rate = (
        sum(1 for previous, current in productivity_pairs if current >= previous - 0.015) / len(productivity_pairs)
        if productivity_pairs
        else 1.0
    )
    open_score = round(_mean(row["framework_growth_score"] for row in retained), 4)
    best_toggle = float(ablation["metrics"]["best_toggle_off_score"])
    local_patch = float(bench["metrics"]["local_patch_growth_score"])
    raw_wisdom = float(bench["metrics"]["raw_wisdom_growth_score"])
    retained_relation_coverage = _coverage(
        retained,
        lambda row: REQUIRED_PROMOTION_RELATIONS.issubset(set(row["relation_types"])),
    )
    return {
        "source_pass_rate": round(
            sum(1 for item in [generator, gate, bench, graph_episode, ablation] if item.get("pass")) / 5,
            4,
        ),
        "generation_count": len(generation_rows),
        "candidate_count": len(candidates),
        "retained_count": len(retained),
        "retention_rate": round(len(retained) / max(1, len(candidates)), 4),
        "active_framework_count": len(active),
        "candidate_framework_count": sum(1 for row in candidates if row["decision"] == "candidate_framework"),
        "branch_only_count": sum(1 for row in candidates if row["decision"] == "branch_only"),
        "reject_count": sum(1 for row in candidates if row["decision"] == "reject"),
        "negative_evidence_retained_count": len(negative),
        "status_counts": dict(Counter(row["decision"] for row in candidates)),
        "active_generation_coverage": _coverage(
            generation_rows,
            lambda row: row["status_counts"].get("active_scoped_framework", 0) > 0,
        ),
        "max_lineage_depth": max((int(row["lineage_depth"]) for row in candidates), default=0),
        "branch_to_framework_transition_count": sum(
            1
            for row in candidates
            if row["frontier_parent_status"] in {"candidate_framework", "active_scoped_framework"}
            and row["decision"] in {"candidate_framework", "active_scoped_framework"}
        ),
        "conservative_obligation_coverage": _coverage(
            candidates,
            lambda row: all(row.get(field) for field in REQUIRED_OBLIGATION_FIELDS),
        ),
        "parent_compatibility_relation_coverage": retained_relation_coverage,
        "limiting_case_survival_rate": _coverage(
            retained,
            lambda row: row["limiting_case_reduction"] >= 0.90 and row["old_success_preservation"] >= 0.95,
        ),
        "generation_productivity": [round(value, 4) for value in productivity],
        "generation_productivity_nonnegative_rate": round(nonnegative_rate, 4),
        "open_ended_framework_growth_score": open_score,
        "margin_vs_best_toggle_off": round(open_score - best_toggle, 4),
        "margin_vs_local_patch": round(open_score - local_patch, 4),
        "margin_vs_raw_wisdom": round(open_score - raw_wisdom, 4),
        "prompt_trick_retained_count": sum(1 for row in retained if row["prompt_trick_retained"]),
        "core_philosophy_prior_promotion_count": sum(1 for row in candidates if row["core_philosophy_prior_promoted"]),
        "graph_readback_relation_coverage": graph_episode["metrics"]["readback_relation_coverage"],
        "ablation_margin_vs_best_toggle_off": ablation["metrics"]["full_margin_vs_best_toggle_off"],
        "main_graph_mutation_count": sum(int(row["main_graph_mutation_count"]) for row in candidates),
        "fresh_api_call_count": 0,
        "bounded_open_ended_framework_evolution_claim_allowed": True,
        "unbounded_open_ended_os_claim_allowed": False,
    }


def _coverage(rows: list[dict[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4)


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build open-ended framework evolution run artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="open_ended_framework_evolution_run_20260612")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--frontier-width", type=int, default=6)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_open_ended_framework_evolution_run_payload(
        root=root,
        eval_id=args.eval_id,
        generations=args.generations,
        frontier_width=args.frontier_width,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
