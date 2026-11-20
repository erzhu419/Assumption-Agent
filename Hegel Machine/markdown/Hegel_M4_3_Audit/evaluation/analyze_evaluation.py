"""Read-only M4.3 evaluation audit: committed receipts and bounded index/gate controls.

No builder/full-study replay, no LLM calls, no changes to repository files.
All in-memory revisions are isolated from the committed snapshots.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get(
        "HEGEL_AUDIT_REPO", "/workspace/scratch/3f14ad2da0ed/repo_m43_346153536/Hegel Machine")))
    parser.add_argument("--out", type=Path, default=Path(os.environ.get(
        "HEGEL_AUDIT_OUT", "/workspace/scratch/3f14ad2da0ed/audit_m43_346153536")) / "evaluation")
    args = parser.parse_args()
    root, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))

    from hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure import (
        CompiledRelationalCueTable,
        build_m4_3_target_suite,
        _find_valid_canonical_role_map_for_box,
        authorize_abstraction_use,
    )
    from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
    from hegel_machine.cognition.m4_multiscale_renormalization import MultiScaleWorkspaceTaskLab

    runner_path = root / "scripts/run_v17_5_m4_3_unified_study.py"
    rspec = importlib.util.spec_from_file_location("m43_audit_runner", runner_path)
    runner = importlib.util.module_from_spec(rspec)
    rspec.loader.exec_module(runner)
    bundle = root / "docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120"
    summary = json.loads((bundle / "part2_m4_3_channel_bound_continual_summary.json").read_text())
    rows = [json.loads(x) for x in (bundle / "part2_m4_3_execution_receipts.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        if not row["is_out_of_scope"]:
            grouped[row["arm"]].append(row)
    key = lambda r: (r["seed"], r["target_id"])
    H = "Hegel_Unified_Continual_Forest"
    N = "Hegel_NoMaintenance_SameHistory"
    hmap = {key(x): x for x in grouped[H]}
    nmap = {key(x): x for x in grouped[N]}
    assert hmap.keys() == nmap.keys()
    pairs = []
    for h in grouped[H]:
        n = nmap[key(h)]
        pairs.append({
            "seed": h["seed"], "target_id": h["target_id"],
            "family": h["mechanism_family"],
            "delta_saved": round(n["cost"]["composite_cost"] - h["cost"]["composite_cost"], 4),
            "H_channel": h["selected_channel"], "N_channel": n["selected_channel"],
            "H_node": h["selected_node_id"], "N_node": n["selected_node_id"],
            "H_candidates": h["evaluated_candidate_count"], "N_candidates": n["evaluated_candidate_count"],
            "H_measurement": h["cost"]["measurement_cost"], "N_measurement": n["cost"]["measurement_cost"],
        })
    fam_deltas = defaultdict(float)
    for pair in pairs:
        fam_deltas[pair["family"]] += pair["delta_saved"]
    delta_total = sum(p["delta_saved"] for p in pairs)
    pi_total = sum(p["delta_saved"] for p in pairs if "PiS" in p["target_id"])
    adi = [p for p in pairs if "Adiabatic" in p["target_id"]]
    quench = [p for p in pairs if "Quench" in p["target_id"]]
    gap = -float(summary["stage2"]["composite_cost"])
    curve = []
    for i, p in enumerate(pairs, 1):
        gap += p["delta_saved"]
        curve.append({"k": i, "target_id": p["target_id"], "seed": p["seed"], "net_maintenance_savings": round(gap, 4)})
    first_cross = next((x["k"] for x in curve if x["net_maintenance_savings"] > 0), None)
    report = {
        "commit": "3461535364fcdb4e9ae9c87ccba502b7e6b56fe8",
        "scope": "Committed 1144 receipts, nominal CostMeter arithmetic; exact event accounting audited separately.",
        "rows": len(rows),
        "unique_keys": len({(r["seed"], r["target_id"], r["arm"]) for r in rows}),
        "arm_counts": dict(Counter(r["arm"] for r in rows)),
        "max_confirm_overlap": max(r["prefix_confirm_overlap_count"] for r in rows),
        "arm_nominal_stats": {
            a: {
                "n": len(rr),
                "pass_count": sum(r["strict_pass"] for r in rr),
                "false_confirmation_count": sum(r["false_confirmation"] for r in rr),
                "stage3_composite_total": round(sum(r["cost"]["composite_cost"] for r in rr), 4),
                "measurement_total": sum(r["cost"]["measurement_cost"] for r in rr),
                "selected_nodes": dict(Counter(r["selected_node_id"] for r in rr)),
            } for a, rr in grouped.items()
        },
        "maintenance": {
            "family_deltas": {f: round(v, 4) for f, v in fam_deltas.items()},
            "stage3_savings": round(delta_total, 4),
            "pi_savings": round(pi_total, 4),
            "pi_fraction": pi_total / delta_total,
            "nonpi_savings": round(delta_total - pi_total, 4),
            "adiabatic_count": len(adi),
            "adiabatic_savings": round(sum(p["delta_saved"] for p in adi), 4),
            "quench_count": len(quench),
            "quench_savings": round(sum(p["delta_saved"] for p in quench), 4),
            "first_cross": first_cross,
            "cross_stays_positive": all(x["net_maintenance_savings"] > 0 for x in curve if x["k"] >= first_cross),
            "adiabatic_positions": [i for i, p in enumerate(pairs, 1) if "Adiabatic" in p["target_id"]],
            "pairs": pairs,
            "curve": curve,
        },
    }

    # Exact current channel-precheck, without costly fitting/replay. Gate provenance is retained.
    forests = {}
    for label, fname in [("stage0", "true_stage0_forest_snapshot.json"),
                         ("unmaintained", "unmaintained_post_stage1_forest_snapshot.json"),
                         ("maintained", "post_stage2_forest_snapshot.json")]:
        forests[label] = runner.load_relocated_forest_from_bundle(bundle / "relocatable_forest_snapshots" / fname, bundle)

    def channel_precheck(forest, target):
        ix = HippoRAGWormholeIndex(forest, damping_alpha=.65, wormhole_equivalence_pooling=True, scale_aware=True)
        ret = ix.retrieve([f"sig:{target.domain_cue}", f"sig:pos_arity_{len(target.feature_names)}"],
                          task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance,
                                     "active_arity": len(target.feature_names), "target_scale_regime": "meso"}, top_k=8)
        trace, chosen = [], None
        for candidate in ret["candidates"]:
            node = candidate["target_node"]
            sm = node.get("scale_map") or {}
            if (sm.get("window_or_partition") or {}).get("scale_regime") != "meso":
                continue
            kids = forest.fine_scale_children_of(node["node_id"])
            in_box, rmap = _find_valid_canonical_role_map_for_box(target.feature_names, node, [*target.pilot_points, *target.confirm_points[:1]])
            gate = authorize_abstraction_use(requested_tier="bounded_trial",
                task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance},
                target_features={"active_arity": len(target.feature_names), "modality": target.modality},
                claim_contract=node["claim_contract"], scale_map=sm, wormhole_edge=candidate.get("wormhole_edge"))
            trace.append({"node_id": node["node_id"], "version": node["version"], "live_children": kids,
                          "in_box": in_box, "role_map": rmap, "gate": gate})
            if kids and in_box and gate["granted"] and chosen is None:
                chosen = node["node_id"]
        return {"selected_channel": "coarse_macro" if chosen else "micro_fine", "authorizing_node": chosen, "trace": trace}

    targets, _ = build_m4_3_target_suite(seed=summary["seeds"][0])
    gate_controls = {}
    for lab, forest in forests.items():
        for label, target in [("adiabatic_hydraulic", targets[20]),
                              ("adiabatic_thermal", replace(targets[20], modality="thermal")),
                              ("quench_thermal", targets[22])]:
            gate_controls[f"{lab}/{label}"] = channel_precheck(forest, target)
    report["channel_precheck_controls"] = gate_controls

    # Real two-channel oracle outputs, without fitting or external API calls.
    channels = []
    for target in (targets[20], targets[22]):
        lab = MultiScaleWorkspaceTaskLab(task_id=target.target_id, surface_domain=target.domain_cue,
            feature_names=target.feature_names, feature_bounds=target.feature_bounds,
            true_scale_regime=target.scale_regime, heldout_points=target.heldout_points,
            tolerance=target.tolerance, micro_fn=target.oracle_fn, coarse_pi_fn=target.coarse_oracle_fn,
            coarse_cost_per_obs=3.0, micro_cost_per_obs=10.0)
        fine = lab.acquire_channel(target.pilot_points, channel="micro_fine", phase="audit_channel_output")
        coarse = lab.acquire_channel(target.pilot_points, channel="coarse_macro", phase="audit_channel_output")
        channels.append({"target_id": target.target_id,
            "rows": [{"features": a["features"], "fine": a["target"], "coarse": b["target"], "abs_difference": abs(a["target"]-b["target"])} for a,b in zip(fine,coarse)]})
    report["actual_two_channel_outputs"] = channels

    # A formal in-memory scope revision does NOT refresh either existing index.
    forest = deepcopy(forests["maintained"])
    node_id = "stage1_W4_monod_shunt"
    ix = HippoRAGWormholeIndex(forest, damping_alpha=.65)
    tab = CompiledRelationalCueTable(forest)
    cc = deepcopy(forest.nodes[node_id]["claim_contract"])
    cc["preconditions"].append({"field": "audit_revision_scope", "op": "==", "value": "new_scope"})
    old_version = forest.nodes[node_id]["version"]
    revised = forest.empirical_revise_node(node_id, witness_episode_id="audit_only_dynamic_index_revision",
        cause="protocol_or_scope_mismatch", updated_claim_contract=cc)
    q = ["sig:internal_monod_shunt_specialization"]
    view = {"active_arity": 2, "goal_kind": "open_workspace_mechanism_prediction", "target_scale_regime": "macro"}
    def query_node(idx):
        matches = [c["target_node"] for c in idx.retrieve(q, task_view=view, top_k=20)["candidates"] if c["target_node"]["node_id"] == node_id]
        return [{"version": n["version"], "preconditions": n["claim_contract"]["preconditions"]} for n in matches]
    report["dynamic_index_control"] = {
        "old_version": old_version, "live_version_after_formal_revision": revised["version"],
        "existing_ppr": query_node(ix), "newly_built_ppr": query_node(HippoRAGWormholeIndex(forest, damping_alpha=.65)),
        "existing_cuetable": query_node(tab), "newly_compiled_cuetable": query_node(CompiledRelationalCueTable(forest)),
        "scope": "Only the index refresh claim; frozen Stage3 evaluation does not modify nodes and is unaffected.",
    }

    # All arms reinitialize the same RNG. This records the actual finite noise sample, not a new experiment.
    noise_values = []
    noise_samples = []
    for seed in summary["seeds"]:
        ts, _ = build_m4_3_target_suite(seed=seed)
        for t in ts:
            if t.pilot_noise_std:
                rng = Random(t.pilot_noise_seed)
                vals = [rng.gauss(0.0, t.pilot_noise_std) for _ in t.pilot_points]
                noise_values.extend(vals)
                noise_samples.append({"seed": seed, "target_id": t.target_id, "pilot_noise_seed": t.pilot_noise_seed,
                    "std_argument": t.pilot_noise_std, "samples": vals})
    report["pilot_noise"] = {
        "generator": "random.Random(task.pilot_noise_seed).gauss(0.0, task.pilot_noise_std)",
        "bounded_distribution": False, "samples": len(noise_values),
        "max_abs_sample": max(abs(x) for x in noise_values),
        "max_abs_sample_over_tolerance": max(abs(x) for x in noise_values) / .062,
        "task_samples": noise_samples,
        "dev_uses_noiseless_oracle": True,
        "dev_indices": [0,1,4,5,8,9,12,13,16,17,20,21],
        "dev_pi_subtype": "adiabatic only; both short-horizon quench members 22,23 absent",
        "fork_comparison_seeds": summary["seeds"][:1],
        "fork_summary_has_accuracy_fields": any("pass" in k or "false" in k for s in summary["fork_noiseless_vs_instrument_noise"].values() for k in s),
    }
    (out / "evaluation_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": report["rows"], "unique_keys": report["unique_keys"],
        "family_deltas": report["maintenance"]["family_deltas"], "pi_fraction": report["maintenance"]["pi_fraction"],
        "adiabatic_savings": report["maintenance"]["adiabatic_savings"], "quench_savings": report["maintenance"]["quench_savings"],
        "first_cross": first_cross, "dynamic_index": report["dynamic_index_control"],
        "gate_channels": {k: v["selected_channel"] for k,v in gate_controls.items()},
        "max_abs_noise": report["pilot_noise"]["max_abs_sample"],
        "output": str(out / "evaluation_analysis.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
