"""Bounded 32-run quality check for the committed 8-onset noise fork.

Reuses committed relocated forest and authored ASTs; never rebuilds formation.
This adds pass/false-confirmation receipts omitted from the official fork summary.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(os.environ.get(
        "HEGEL_AUDIT_REPO", "/workspace/scratch/3f14ad2da0ed/repo_m43_346153536/Hegel Machine")))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get(
        "HEGEL_AUDIT_OUT", "/workspace/scratch/3f14ad2da0ed/audit_m43_346153536")) / "evaluation" / "fork_quality")
    a = ap.parse_args()
    root, out = a.repo.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    from hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure import (
        build_m4_3_target_suite, evaluate_m4_3_target_under_arm,
        make_atomic_operator_spec, apply_structural_combinator,
    )
    from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
    spec = importlib.util.spec_from_file_location("m43_fork_runner", root / "scripts/run_v17_5_m4_3_unified_study.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    bundle = root / "docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120"
    forest = runner.load_relocated_forest_from_bundle(bundle / "relocatable_forest_snapshots/post_stage2_forest_snapshot.json", bundle)
    # Exact order/spec construction from M42 builder lines 812-880 and 1897-1911.
    history = [
        ("op_atomic_monod_2d", make_atomic_operator_spec("michaelis_1d", arity=2)),
        ("op_atomic_murray_1d", make_atomic_operator_spec("murray_power_2_3_1d")),
        ("op_atomic_exp_relax_2d", make_atomic_operator_spec("exp_relax_2d")),
        ("op_atomic_linear_1d", make_atomic_operator_spec("linear_1d")),
        ("op_atomic_michaelis_1d", make_atomic_operator_spec("michaelis_1d")),
        ("op_lure_variational_cellular_2d", make_atomic_operator_spec("variational_cellular_2d")),
        ("op_meso_schur_pi_s", apply_structural_combinator(make_atomic_operator_spec("exp_relax_2d"), "additive_secondary_shunt", secondary_role=0)),
    ]
    for tag, nid in [("op_composed_haldane_shunt_2d", "stage1_W1_haldane_shunt"),
                     ("op_composed_osc_shunt_4p_2d", "stage1_W2_osc_shunt_4p"),
                     ("op_composed_murray_capped_1d", "stage1_W3_murray_capped"),
                     ("op_composed_monod_shunt_2d", "stage1_W4_monod_shunt"),
                     ("op_composed_murray_shunt_2d", "stage1_W5_murray_shunt")]:
        art_path = Path(forest.nodes[nid]["claim_contract"]["conclusion_or_action"]["authored_artifact_path"])
        history.append((tag, deepcopy(json.loads(art_path.read_text())["operator_spec"])))
    (out / "reused_ordinary_history_specs.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    all_rows, results = [], {}
    seed = 20301120
    for label, sigma in [("noiseless_sigma_0", 0.0), ("bounded_instrument_sigma_0_0020", .002)]:
        targets, _ = build_m4_3_target_suite(seed=seed, onset_pilot_noise_std=sigma)
        targets = [t for t in targets if t.mechanism_family.startswith("OnsetWalk_")]
        f = deepcopy(forest)
        ix = HippoRAGWormholeIndex(f, damping_alpha=.65, wormhole_equivalence_pooling=True, scale_aware=True)
        for t in targets:
            for name, pol in [("Hegel_PPR", "first_acceptable"), ("BestPilotRMSE", "min_pilot_rmse")]:
                is_h = name == "Hegel_PPR"
                row = evaluate_m4_3_target_under_arm(t, arm_name=name, seed=seed,
                    work_dir=out / "workspaces" / label / name / t.target_id,
                    forest=f if is_h else None, index=ix if is_h else None,
                    ordinary_history_specs=None if is_h else history,
                    fallback_pool_specs=history, selection_policy=pol)
                row["noise_regime"] = label
                all_rows.append(row)
                with (out / "fork_quality_receipts.jsonl").open("a") as fp:
                    fp.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps({"regime": label, "arm": name, "target": t.target_id,
                    "strict_pass": row["strict_pass"], "false_confirmation": row["false_confirmation"],
                    "C": row["cost"]["composite_cost"]}), flush=True)
        results[label] = {}
        for name in ("Hegel_PPR", "BestPilotRMSE"):
            rr = [r for r in all_rows if r["noise_regime"] == label and r["arm"] == name]
            results[label][name] = {"count": len(rr), "strict_pass_count": sum(r["strict_pass"] for r in rr),
                "false_confirmation_count": sum(r["false_confirmation"] for r in rr),
                "mean_meas": mean(r["cost"]["measurement_cost"] for r in rr),
                "mean_composite": round(mean(r["cost"]["composite_cost"] for r in rr), 4),
                "max_confirm_overlap": max(r["prefix_confirm_overlap_count"] for r in rr)}
    official = json.loads((bundle / "part2_m4_3_channel_bound_continual_summary.json").read_text())["fork_noiseless_vs_instrument_noise"]
    checks = {}
    for regime, arms in results.items():
        for name, rr in arms.items():
            checks[f"{regime}/{name}"] = rr["mean_composite"] == official[regime][f"{name}_mean_composite"] and rr["mean_meas"] == official[regime][f"{name}_mean_meas"]
    final = {"scope": "First official seed, identical 8 onset tasks, 2 noise regimes, 2 arms; quality check only, no tuning.",
        "results": results, "cost_means_match_official": checks}
    (out / "fork_quality_summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
