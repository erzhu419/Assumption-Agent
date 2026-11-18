#!/usr/bin/env python3
"""Recompute V13 saved-receipt outcomes, without running experiments or models.

Example:
    python recompute_v13.py --bundle /path/to/extracted/handoff \
        --output recomputed_results.json

The bundle must contain receipts/v13_evidence_recursion_and_routing_20261117/.
Uses only the Python standard library. Success-pair subsets, sign-flip tests,
and maintenance-prefix arithmetic are descriptive diagnostics, not causal
estimates or replacements for a missing experimental arm.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

RECEIPT_RELATIVE_PATH = Path(
    "receipts/v13_evidence_recursion_and_routing_20261117/"
    "pr_c_tightened_source_paired_study_20261117.json"
)
TOLERANCE = 1e-7
TARGET_BUDGETS = {
    "compute_cost": 680000,
    "observation_cost": 620,
    "measurement_cost": 12400,
}
SOURCE_BUDGETS = {
    "compute_cost": 740000,
    "observation_cost": 660,
    "measurement_cost": 13200,
}
COLD = "K_cold"
ORDINARY = "R_ordinary_optimized_lazy"
NO_H = "K_routing_no_H_star"
PLUS_H = "K_routing_plus_H_star"
ARM_NAMES = (COLD, ORDINARY, NO_H, PLUS_H)


def check_and_recompute_gates(
    episode: dict[str, Any],
    budgets: dict[str, int],
    mismatches: list[dict[str, Any]],
) -> None:
    """Compare stored booleans, then use recomputed values for all aggregates."""
    costs = episode["measured_costs"]
    confirmed = bool(
        episode["status"] == "completed"
        and episode["confirmation_loss"] is not None
        and episode["confirmation_loss"] <= TOLERANCE
    )
    within = all(costs[key] <= bound for key, bound in budgets.items())
    external_ok = episode["external_fresh48_mse"] <= TOLERANCE
    calculated = {
        "internally_confirmed": confirmed,
        "strict_pass": confirmed and external_ok and within,
        "false_internal_confirmation": confirmed and not external_ok,
        "within_predeclared_budget": within,
    }
    for key, value in calculated.items():
        if episode[key] != value:
            mismatches.append({
                "episode": episode["episode_tag"],
                "arm": episode["arm"],
                "field": key,
                "reported": episode[key],
                "recomputed": value,
            })
        episode[key] = value


def sign_flip_p(values: list[float]) -> float:
    nonzero = [value for value in values if value]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    exceedances = sum(
        abs(sum(sign * abs(value) for sign, value in zip(signs, nonzero)))
        >= observed - 1e-10
        for signs in product((-1, 1), repeat=len(nonzero))
    )
    return exceedances / (2 ** len(nonzero))


def recompute(bundle: Path) -> dict[str, Any]:
    receipt = bundle / RECEIPT_RELATIVE_PATH
    receipt_bytes = receipt.read_bytes()
    data = json.loads(receipt_bytes)
    chains = data["unexposed_evaluation_chains"]
    result: dict[str, Any] = {
        "receipt_relative_path": str(RECEIPT_RELATIVE_PATH),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "scope": "Independent arithmetic recheck of saved receipts; no experiments rerun",
        "clusters": len(chains),
        "tolerance": TOLERANCE,
        "target_budgets": TARGET_BUDGETS,
        "source_budgets": SOURCE_BUDGETS,
        "arms": {},
        "comparisons": {},
        "gate_mismatches": [],
        "source_failures": [],
        "statistical_scope": (
            "Source-cluster paired sign flips are exploratory, assume independent "
            "clusters and sign exchangeability, and are not adjusted for multiple "
            "comparisons. They do not remove policy confounding or establish equivalence."
        ),
    }

    for arm in ARM_NAMES:
        episodes = [ep for chain in chains for ep in chain["arms_target_results"][arm]]
        for episode in episodes:
            check_and_recompute_gates(episode, TARGET_BUDGETS, result["gate_mismatches"])
        aggregate = {
            "n": len(episodes),
            "strict_passes": sum(ep["strict_pass"] for ep in episodes),
            "false_confirmations": sum(ep["false_internal_confirmation"] for ep in episodes),
            "internally_completed": sum(ep["internally_confirmed"] for ep in episodes),
            "within_budget_all": all(ep["within_predeclared_budget"] for ep in episodes),
            "mean_compute": mean(ep["measured_costs"]["compute_cost"] for ep in episodes),
            "total_compute": sum(ep["measured_costs"]["compute_cost"] for ep in episodes),
            "mean_observations": mean(ep["measured_costs"]["observation_cost"] for ep in episodes),
            "mean_wall_seconds": mean(ep["measured_costs"]["wall_seconds"] for ep in episodes),
            "reconstruction_compute_total": sum(
                ep["ordinary_lazy_episode_reconstruction_compute"] for ep in episodes
            ),
            "first_cache_hits": sum(
                ep["ordinary_lazy_first_call_cache_hit"] is True for ep in episodes
            ),
            "by_role": {},
            "failures": [],
        }
        for role in sorted({ep["target_regime_role"] for ep in episodes}):
            selected = [ep for ep in episodes if ep["target_regime_role"] == role]
            aggregate["by_role"][role] = {
                "passes": sum(ep["strict_pass"] for ep in selected),
                "mean_compute": mean(ep["measured_costs"]["compute_cost"] for ep in selected),
                "mean_observations": mean(ep["measured_costs"]["observation_cost"] for ep in selected),
            }
        failure_fields = (
            "episode_tag", "status", "confirmation_loss", "external_fresh48_mse",
            "false_internal_confirmation", "measured_costs", "selected_kind",
        )
        aggregate["failures"] = [
            {key: ep[key] for key in failure_fields}
            for ep in episodes if not ep["strict_pass"]
        ]
        result["arms"][arm] = aggregate

    comparisons = ((NO_H, ORDINARY), (PLUS_H, ORDINARY), (PLUS_H, NO_H), (NO_H, COLD))
    for focal, reference in comparisons:
        success_differences = []
        compute_savings = []
        changed_cases = []
        both_success_savings = []
        prefix_diagnostic = []
        # This accounting illustration applies only to eager-no-H versus lazy-R.
        # Do not subtract maintenance from comparisons where both arms share it.
        include_prefix_diagnostic = (focal, reference) == (NO_H, ORDINARY)
        for chain in chains:
            focal_episodes = chain["arms_target_results"][focal]
            reference_episodes = chain["arms_target_results"][reference]
            if len(focal_episodes) != len(reference_episodes):
                raise ValueError("Cannot pair arms with different episode counts")
            pairs = list(zip(focal_episodes, reference_episodes))
            success_differences.append(sum(
                int(fe["strict_pass"]) - int(re["strict_pass"]) for fe, re in pairs
            ))
            compute_savings.append(mean(
                re["measured_costs"]["compute_cost"] - fe["measured_costs"]["compute_cost"]
                for fe, re in pairs
            ))
            cumulative = -chain["source_formation_measured_ledger"][
                "reconciliation_subset_compute_measured"
            ] if include_prefix_diagnostic else 0
            prefix_values = []
            for fe, re in pairs:
                if fe["episode_tag"] != re["episode_tag"]:
                    raise ValueError("Episode order does not match between paired arms")
                delta = re["measured_costs"]["compute_cost"] - fe["measured_costs"]["compute_cost"]
                if include_prefix_diagnostic:
                    cumulative += delta
                    prefix_values.append(cumulative)
                if fe["strict_pass"] and re["strict_pass"]:
                    both_success_savings.append(delta)
                if fe["strict_pass"] != re["strict_pass"]:
                    changed_cases.append({
                        "episode": fe["episode_tag"],
                        "role": fe["target_regime_role"],
                        "focal_pass": fe["strict_pass"],
                        "reference_pass": re["strict_pass"],
                        "focal_compute": fe["measured_costs"]["compute_cost"],
                        "reference_compute": re["measured_costs"]["compute_cost"],
                        "focal_external_mse": fe["external_fresh48_mse"],
                        "reference_external_mse": re["external_fresh48_mse"],
                    })
            if include_prefix_diagnostic:
                prefix_diagnostic.append({
                    "chain_id": chain["chain_id"],
                    "prefix_net_compute_savings_after_source_maintenance": prefix_values,
                })
        comparison = {
            "success_differences_per_cluster": success_differences,
            "target_compute_savings_per_cluster_mean": compute_savings,
            "paired_success_signflip_p": sign_flip_p(success_differences),
            "paired_compute_signflip_p_descriptive": sign_flip_p(compute_savings),
            "mean_compute_saving": mean(compute_savings),
            "median_cluster_mean_compute_saving": median(compute_savings),
            "compute_saving_fraction": mean(compute_savings) / result["arms"][reference]["mean_compute"],
            "same_success_pairs": len(both_success_savings),
            "same_success_pairs_mean_compute_saving": (
                mean(both_success_savings) if both_success_savings else None
            ),
            "same_success_pairs_interpretation": (
                "Descriptive subset selected after outcomes are observed; not a causal "
                "estimate, an independent robustness test, or a quality-adjusted benefit."
            ),
            "changed_cases": changed_cases,
        }
        if include_prefix_diagnostic:
            comparison["arithmetic_prefix_cost_with_maintenance_assigned_only_to_focal"] = prefix_diagnostic
            comparison["prefix_accounting_scope"] = (
                "Illustration only for K_no_H versus R: assigns measured reconciliation "
                "maintenance to eager K while conditioning on shared raw source history. "
                "This is not a measured end-to-end counterfactual and does not remove "
                "the H-star scheduling confound or include extra acquisition costs."
            )
        result["comparisons"][focal + "__vs__" + reference] = comparison

    sources = [ep for chain in chains for ep in chain["source_episodes"]]
    anchors = [chain["calibration_anchor_episode"] for chain in chains]
    source_failure_fields = (
        "episode_tag", "status", "confirmation_loss", "external_fresh48_mse",
        "strict_pass", "measured_costs",
    )
    for episode in sources + anchors:
        check_and_recompute_gates(episode, SOURCE_BUDGETS, result["gate_mismatches"])
        if not episode["strict_pass"]:
            result["source_failures"].append({key: episode[key] for key in source_failure_fields})
    result["sources"] = {
        "ordinary_source_episodes": len(sources),
        "internally_confirmed": sum(ep["internally_confirmed"] for ep in sources),
        "strict_passes": sum(ep["strict_pass"] for ep in sources),
        "false_confirmations": sum(ep["false_internal_confirmation"] for ep in sources),
        "neutral_anchor_episodes": len(anchors),
        "neutral_anchor_strict_passes": sum(ep["strict_pass"] for ep in anchors),
        "mean_source_compute": sum(ep["measured_costs"]["compute_cost"] for ep in sources) / len(chains),
        "mean_maintenance_compute": mean(
            chain["source_formation_measured_ledger"]["reconciliation_subset_compute_measured"]
            for chain in chains
        ),
        "mean_anchor_compute": mean(ep["measured_costs"]["compute_cost"] for ep in anchors),
        "total_maintenance_compute": sum(
            chain["source_formation_measured_ledger"]["reconciliation_subset_compute_measured"]
            for chain in chains
        ),
    }
    result["policy_confound"] = {
        ORDINARY: "prioritize_generalized_families = True",
        NO_H: "prioritize_generalized_families = False",
        PLUS_H: "prioritize_generalized_families = True",
        "note": (
            "R versus K_no_H changes both organization/reconstruction and H-star "
            "scheduling; not a clean attribution to learned routing."
        ),
    }
    base = result["arms"][ORDINARY]
    focal = result["arms"][NO_H]
    savings = base["total_compute"] - focal["total_compute"]
    result["cost_decomposition"] = {
        "total_observed_target_compute_saving": savings,
        "ordinary_reconstruction_compute": base["reconstruction_compute_total"],
        "fraction_direct_reconstruction": base["reconstruction_compute_total"] / savings if savings else None,
        "remaining_nonreconstruction_difference": savings - base["reconstruction_compute_total"],
        "maintenance_only_mean_breakeven_ratio": (
            result["sources"]["mean_maintenance_compute"] / (savings / focal["n"])
            if savings > 0 else None
        ),
        "not_a_valid_causal_decomposition": (
            "Cannot subtract reconstruction costs to create an unrun baseline; policies "
            "differ and budgets alter paths. The ratio assumes constant workload-average "
            "savings although the ordinary cache is warm after the first target."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Extracted V13 handoff directory")
    parser.add_argument("--output", type=Path, default=Path("recomputed_results.json"), help="Output JSON path")
    args = parser.parse_args()
    result = recompute(args.bundle.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "written": str(output),
        "receipt_sha256": result["receipt_sha256"],
        "gate_mismatches": result["gate_mismatches"],
        "source_strict_passes": result["sources"]["strict_passes"],
        "arm_strict_passes": {arm: value["strict_passes"] for arm, value in result["arms"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
