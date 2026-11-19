"""Execute Preregistered Hegel V17.1 (M3.1) Audit Closure, Retrieval Factorization & Walk Study.

Writes full static evidence bundle to:
  `docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119/`
without modifying any prior V16.8, V16.9, or V17.0 evidence directories!
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
from tempfile import TemporaryDirectory
import time
from typing import Any

from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m3_1_walk_factorization import (
    M3_1FactorizedResearcher,
    build_m3_1_high_load_confirmation_point,
    build_m3_1_pilot_points,
    extract_m3_1_observable_signatures,
    make_m3_1_factorized_walk_suite,
)
from hegel_machine.cognition.m3_scale_universe import (
    M3ForestResearcher,
    M3HExACodeSkills,
    M3StrongOrdinaryAllPriors,
    make_m3_large_scale_suite,
)
import hegel_machine.cognition.m3_scale_universe as m3_mod
from hegel_machine.cognition.open_python_revision import run_open_revision_chain_experiment
from hegel_machine.cognition.wormhole_forest import TwoStageWormholeGate


REPO_ROOT = Path(__file__).resolve().parents[1]
EVID_DIR = REPO_ROOT / "docs" / "evidence" / "v17_1_m3_1_audit_closure_and_walk_20261119"

DEV_SEEDS = [("m3_1_dev_01", 8_028_001), ("m3_1_dev_02", 8_028_002)]
EVAL_SEEDS = [
    ("m3_1_eval_01", 8_028_101),
    ("m3_1_eval_02", 8_028_202),
    ("m3_1_eval_03", 8_028_303),
    ("m3_1_eval_04", 8_028_404),
]
OPEN_REV_SEEDS = [5_026_101, 5_026_202, 5_026_303, 5_026_404]

FACTORIZED_FOREST_MODES = [
    ("Hegel_M3_1_HippoPPR_Walk_Pool", "hipporag_ppr"),
    ("HippoPPR_Alpha0_WormholePool", "hipporag_alpha0_pool"),
    ("HippoPPR_Walk_NoPool", "hipporag_walk_nopool"),
    ("Flat_BM25_WormholePool", "flat_bm25_wormhole_pool"),
    ("Flat_BM25_EqualMetadata", "flat_bm25"),
    ("Flat_Cosine_EqualMetadata", "flat_cosine"),
    ("Forest_VerticalPPR_NoWormholes", "vertical_ppr_no_wh"),
    ("HippoPPR_UncheckedRecall", "hipporag_unchecked"),
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _infer_truth_schema(tname: str) -> str | None:
    exact_map = {
        "T01_assoc_inverted_capped_datacenter": "liebig_capped_min_2d",
        "T02_assoc_inverted_ringing_lc_bus": "underdamped_ringing_2d",
        "T03_assoc_inverted_warm_battery_relax": "conjugate_relaxation_state_memory",
        "T04_assoc_inverted_photobioreactor_shunt": "haldane_shunt_2d",
        "T05_assoc_reordered_tcp_backpressure": "conjugate_relaxation_reset",
        "T06_assoc_inverted_hill_riboswitch": "hill_cooperative_n2_1d",
        "T07_hub_harmonic_heat_exchanger": "parallel_harmonic_flux_2d",
        "T08_hub_battery_liebig_transport": "liebig_min_2d",
        "T09_hub_overdamped_pneumatic_valve": "overdamped_charging_2d",
        "T10_hub_cobb_douglas_wireless_mimo": "cobb_douglas_sqrt_2d",
        "T11_hub_exp_washout_aerosol": "exp_washout_decay_1d",
        "T12_hub_damped_tank_with_seepage": "damped_driven_with_leak_3d",
        "T13_guard_two_hop_strict_rollover": "congestion_rollover_1d",
        "T14_guard_micro_crest_scale_block": "inv_quad_crest_1d",
        "T15_guard_macro_window_mean_allow": "rational_monod_1d",
        "T16_guard_capped_exclude_uncapped": "liebig_capped_min_2d",
        "T17_guard_ringing_exclude_overdamped": "underdamped_ringing_2d",
        "T18_guard_bimodal_exclude_1mode": "bimodal_relaxation_2mode",
        "T19_simple_linear_hall_probe": "linear_proportional_1d",
        "T20_simple_monod_membrane_flux": "rational_monod_1d",
        "T21_simple_hill_cooperative_sensor": "hill_cooperative_n2_1d",
        "T22_simple_relaxation_hydraulic_damper": "conjugate_relaxation_reset",
        "T23_simple_tri_liebig_fertilizer": "tri_resource_liebig_min_3d",
        "T24_simple_inverted_quenching_monod": "rational_monod_1d",
        "T25_walk_safe_pilot_capped_grid": "liebig_capped_min_2d",
        "T26_walk_safe_pilot_pore_rollover": "congestion_rollover_1d",
        "T27_walk_safe_pilot_haldane_fermentor": "haldane_shunt_2d",
        "T28_walk_safe_pilot_ringing_converter": "underdamped_ringing_2d",
    }
    return exact_map.get(tname)


def _run_with_m3_1_hooks(fn):
    orig_p = m3_mod.build_m3_pilot_points
    orig_c = m3_mod.build_m3_confirmation_point
    orig_e = m3_mod.extract_m3_observable_signatures
    m3_mod.build_m3_pilot_points = build_m3_1_pilot_points
    m3_mod.build_m3_confirmation_point = build_m3_1_high_load_confirmation_point
    m3_mod.extract_m3_observable_signatures = extract_m3_1_observable_signatures
    try:
        return fn()
    finally:
        m3_mod.build_m3_pilot_points = orig_p
        m3_mod.build_m3_confirmation_point = orig_c
        m3_mod.extract_m3_observable_signatures = orig_e


def run_part1_closure_verification(work_root: Path) -> dict[str, Any]:
    """Verify all M3 audit closure controls on the updated codebase."""
    suite_m3 = make_m3_large_scale_suite(7_027_404)
    h_m3 = M3ForestResearcher(retrieval_mode="hipporag_ppr")
    logs_m3 = h_m3.form_and_revise_36_episodes("closure_check_eval04", suite_m3, work_root / "m3_closure")
    ep_logs = [r for r in logs_m3 if r["phase"] in ("source", "source_nearmiss")]
    verified_count = sum(1 for r in ep_logs if r["skeleton_status"] == "VERIFIED_PURE_AST")
    bridge_logged = any(r["phase"] == "source_bridge_calibration" for r in logs_m3)

    # Check node_id rename invariance across all 24 standard targets on `eval04`
    renamed_forest = deepcopy(h_m3.forest)
    remap = {nid: f"zz_renamed_{i:03d}" for i, nid in enumerate(reversed(sorted(renamed_forest.nodes)))}
    renamed_forest.nodes = {
        remap[k]: {**deepcopy(v), "node_id": remap[k]} for k, v in renamed_forest.nodes.items()
    }
    for ve in renamed_forest.local_implication_edges:
        ve["stronger_node_id"] = remap[ve["stronger_node_id"]]
        ve["weaker_node_id"] = remap[ve["weaker_node_id"]]
    for wh in renamed_forest.horizontal_wormhole_edges.values():
        wh["node_a_id"] = remap[wh["node_a_id"]]
        wh["node_b_id"] = remap[wh["node_b_id"]]

    idx_orig_ppr = HippoRAGWormholeIndex(h_m3.forest)
    idx_ren_ppr = HippoRAGWormholeIndex(renamed_forest)
    idx_orig_bm25 = FlatEqualMetadataIndex(h_m3.forest, mode="bm25")
    idx_ren_bm25 = FlatEqualMetadataIndex(renamed_forest, mode="bm25")
    sample_q = ["sig:arity_2", "sig:calibrated_inverted_port", "sig:calibrated_shifted_offset", "sig:2d_multi_factor_positive", "sig:2d_symmetric_axes", "sig:high_corner_sublinear"]
    tv = {"goal_kind": "open_workspace_mechanism_prediction", "active_arity": 2}
    ppr_same = (
        [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in idx_orig_ppr.retrieve(sample_q, task_view=tv, top_k=6)["candidates"]]
        == [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in idx_ren_ppr.retrieve(sample_q, task_view=tv, top_k=6)["candidates"]]
    )
    bm25_same = (
        [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in idx_orig_bm25.retrieve(sample_q, task_view=tv, top_k=6)["candidates"]]
        == [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in idx_ren_bm25.retrieve(sample_q, task_view=tv, top_k=6)["candidates"]]
    )

    return {
        "m3_total_episodes": len(ep_logs),
        "m3_verified_pure_ast_count": verified_count,
        "m3_mismatch_count": len(ep_logs) - verified_count,
        "m3_horizontal_wormholes_formed": h_m3.index.edge_counts["horizontal_wormhole_edges"] if h_m3.index else 0,
        "bridge_fit_logged_in_source_episodes": bridge_logged,
        "identifier_independent_tiebreak_ppr": ppr_same,
        "identifier_independent_tiebreak_bm25": bm25_same,
    }


def run_part2_factorized_walk_benchmark(work_root: Path) -> dict[str, Any]:
    """Run 2-DEV gate calibration + 4-EVAL factorized benchmark across all 10 arms!"""
    gate = TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=12)
    dev_rollout_logs: list[dict[str, Any]] = []

    for dev_cid, dev_seed in DEV_SEEDS:
        d_suite = make_m3_1_factorized_walk_suite(dev_seed, target_k=36)
        d_res = M3_1FactorizedResearcher(gate=gate, retrieval_mode="hipporag_ppr", damping_alpha=0.65)
        d_res.form_and_revise_episodes(dev_cid, d_suite, work_root / dev_cid / "src")
        for _cat, tname, (tfac, tev) in d_suite["targets"]:
            obs_roll = d_res.run_paired_post_pilot_rollout(tfac, tev, work_root / dev_cid / "roll" / tname)
            if obs_roll is not None:
                gate.observe_paired_dev_rollout(
                    obs_roll["signature_key"],
                    transfer_remaining_cost=obs_roll["transfer_remaining_cost"],
                    direct_remaining_cost=obs_roll["direct_remaining_cost"],
                    transfer_quality_ok=obs_roll["transfer_quality_ok"],
                    direct_quality_ok=obs_roll["direct_quality_ok"],
                )
                dev_rollout_logs.append({"dev_cluster": dev_cid, "target": tname, **obs_roll})

    frozen_gate = deepcopy(gate)
    frozen_gate.uncovered_exploration_calls_used = 0

    source_episodes: list[dict[str, Any]] = []
    target_episodes: list[dict[str, Any]] = []
    oos_episodes: list[dict[str, Any]] = []
    forest_snapshots: dict[str, Any] = {}

    for cid, cseed in EVAL_SEEDS:
        c_dir = work_root / cid
        suite = make_m3_1_factorized_walk_suite(cseed, target_k=36)

        # 1. Form shared frozen forest ONCE per cluster -> clone identically across all 8 forest arms!
        primary = M3_1FactorizedResearcher(gate=deepcopy(frozen_gate), retrieval_mode="hipporag_ppr", damping_alpha=0.65)
        src_h = primary.form_and_revise_episodes(cid, suite, c_dir / "forest_src")
        source_episodes.extend(src_h)

        forest_arms: dict[str, M3_1FactorizedResearcher] = {}
        for arm_label, mode_code in FACTORIZED_FOREST_MODES:
            cloned = primary.clone_with_mode(mode_code)
            cloned.gate = deepcopy(frozen_gate)
            forest_arms[arm_label] = cloned
            if arm_label != "Hegel_M3_1_HippoPPR_Walk_Pool":
                source_episodes.append({
                    "phase": "index_build",
                    "arm": arm_label,
                    "cluster_id": cid,
                    "episode_name": f"index_build_{mode_code}",
                    "surface_domain": "forest_index",
                    "strict_pass": True,
                    "costs": {
                        "observation_cost": 0,
                        "measurement_cost": 0.0,
                        "compute_cost": cloned.index_build_compute_cost,
                        "composite_cost": round(0.05 * cloned.index_build_compute_cost, 4),
                        "measured_python_wall_seconds": round(cloned.index_build_wall_seconds, 5),
                    },
                })

        forest_snapshots[cid] = {
            "edge_counts": primary.index.edge_counts if primary.index else {},
            "revision_events": primary.revision_events,
            "learned_scale_certificates": primary.learned_scale_certificates,
        }

        # 2. Form non-forest baselines (`O_strong_AllValidatedPriors` & `O_HExA_CodeSkills`) under identical hooks
        ord_strong = M3StrongOrdinaryAllPriors()
        src_o = _run_with_m3_1_hooks(lambda: ord_strong.run_36_episode_source_phase(suite, c_dir / "ord_src"))
        for r in src_o:
            r["cluster_id"] = cid
        source_episodes.extend(src_o)

        hexa = M3HExACodeSkills()
        src_hx = _run_with_m3_1_hooks(lambda: hexa.run_36_episode_source_phase(suite, c_dir / "hexa_src"))
        for r in src_hx:
            r["cluster_id"] = cid
        source_episodes.extend(src_hx)

        # 3. Evaluate all 26 unseen targets across all 10 arms!
        for cat, tname, (tfac, tev) in suite["targets"]:
            truth_tag = _infer_truth_schema(tname)
            for arm_label, researcher in forest_arms.items():
                lb = tfac()
                ws = c_dir / "targets" / arm_label / tname
                sub = researcher.solve_target(
                    tname, lb, ws, ground_truth_schema_for_audit_log_only=truth_tag
                )
                ev = tev(ws, sub, lb.observations)
                rlog = researcher.retrieval_logs[-1]
                raw_top = rlog.get("raw_top_schemas", [])
                target_episodes.append({
                    "cluster_id": cid,
                    "category": cat,
                    "target_name": tname,
                    "arm": arm_label,
                    "strict_pass": bool(ev["strict_pass"]),
                    "false_confirmation": bool(ev["false_confirmation"]),
                    "eval_rmse": ev["rmse"],
                    "selected_schema": sub.get("selected_schema"),
                    "truth_schema_audit_only": truth_tag,
                    "recall_at_1_raw": bool(truth_tag and raw_top[:1] == [truth_tag]),
                    "recall_at_2_raw": bool(truth_tag and truth_tag in raw_top[:2]),
                    "confirmed_via_recall": bool(rlog.get("confirmed_via_recall", False)),
                    "selected_reached_only_by_walk": bool(rlog.get("selected_reached_only_by_walk", False)),
                    "non_direct_walk_recalled_count": int(rlog.get("non_direct_walk_recalled_count", 0)),
                    "wrong_candidate_trials": int(rlog.get("wrong_candidate_trials", 0)),
                    "costs": {
                        **lb.meter.snapshot(),
                        "measured_python_wall_seconds": round(lb.measured_python_wall_seconds, 4),
                    },
                })

            for non_f_label, non_f_obj in [
                ("O_strong_AllValidatedPriors", ord_strong),
                ("O_HExA_CodeSkills", hexa),
            ]:
                lb = tfac()
                ws = c_dir / "targets" / non_f_label / tname
                sub = _run_with_m3_1_hooks(lambda obj=non_f_obj, l=lb, w=ws, tn=tname: obj.solve_target(tn, l, w))
                ev = tev(ws, sub, lb.observations)
                target_episodes.append({
                    "cluster_id": cid,
                    "category": cat,
                    "target_name": tname,
                    "arm": non_f_label,
                    "strict_pass": bool(ev["strict_pass"]),
                    "false_confirmation": bool(ev["false_confirmation"]),
                    "eval_rmse": ev["rmse"],
                    "selected_schema": sub.get("selected_schema"),
                    "truth_schema_audit_only": truth_tag,
                    "recall_at_1_raw": False,
                    "recall_at_2_raw": False,
                    "confirmed_via_recall": False,
                    "selected_reached_only_by_walk": False,
                    "non_direct_walk_recalled_count": 0,
                    "wrong_candidate_trials": 0,
                    "costs": {
                        **lb.meter.snapshot(),
                        "measured_python_wall_seconds": round(lb.measured_python_wall_seconds, 4),
                    },
                })

        # 4. Out-of-scope honesty controls (`OOS01`, `OOS02`)
        for oname, (ofac, oev) in suite["out_of_scope_controls"]:
            lb_o = ofac()
            ws_o = c_dir / "oos" / oname
            sub_o = forest_arms["Hegel_M3_1_HippoPPR_Walk_Pool"].solve_target(oname, lb_o, ws_o)
            ev_o = oev(ws_o, sub_o, lb_o.observations)
            oos_episodes.append({
                "cluster_id": cid,
                "target_name": oname,
                "submission_status": sub_o["status"],
                "strict_pass": ev_o["strict_pass"],
                "false_confirmation": ev_o["false_confirmation"],
                "confirmation_rmse": sub_o.get("confirmation_rmse"),
            })

    return {
        "dev_gate_calibration": {
            "paired_dev_records": frozen_gate.paired_dev_records,
            "rollout_count": len(dev_rollout_logs),
            "rollouts": dev_rollout_logs,
        },
        "forest_snapshots": forest_snapshots,
        "source_episodes": source_episodes,
        "target_episodes": target_episodes,
        "oos_episodes": oos_episodes,
    }


def run_part3_isolated_k_scaling(work_root: Path) -> dict[str, Any]:
    """Single-factor isolated-K distractor scaling (`K in {20, 36, 52}`) on fixed associative + walk targets!"""
    k_values = (20, 36, 52)
    seeds = (8_028_101, 8_028_202)
    modes = [
        ("Hegel_M3_1_HippoPPR_Walk_Pool", "hipporag_ppr"),
        ("HippoPPR_Alpha0_WormholePool", "hipporag_alpha0_pool"),
        ("Flat_BM25_WormholePool", "flat_bm25_wormhole_pool"),
        ("Flat_BM25_EqualMetadata", "flat_bm25"),
        ("Forest_VerticalPPR_NoWormholes", "vertical_ppr_no_wh"),
    ]
    rows: list[dict[str, Any]] = []

    for k_val in k_values:
        for sd in seeds:
            suite = make_m3_1_factorized_walk_suite(sd, target_k=k_val)
            gate = TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=20)
            base = M3_1FactorizedResearcher(gate=gate, retrieval_mode="hipporag_ppr", damping_alpha=0.65)
            base.form_and_revise_episodes(f"iso_k{k_val}_{sd}", suite, work_root / f"k{k_val}_{sd}" / "src")
            arms = {lbl: base.clone_with_mode(mc) for lbl, mc in modes}

            # Evaluate the fixed 10 multi-tree associative + walk targets (`T01..T06` + `T25..T28`)
            focus_targets = [t for t in suite["targets"] if t[0] in {"T_Assoc", "T_WalkBridge"}]
            for cat, tname, (tfac, tev) in focus_targets:
                truth_tag = _infer_truth_schema(tname)
                for lbl, ag in arms.items():
                    lb = tfac()
                    ws = work_root / f"k{k_val}_{sd}" / lbl / tname
                    sub = ag.solve_target(tname, lb, ws, ground_truth_schema_for_audit_log_only=truth_tag)
                    ev = tev(ws, sub, lb.observations)
                    rlog = ag.retrieval_logs[-1]
                    rows.append({
                        "K": k_val,
                        "seed": sd,
                        "category": cat,
                        "target": tname,
                        "arm": lbl,
                        "strict_pass": bool(ev["strict_pass"]),
                        "confirmed_via_recall": bool(rlog["confirmed_via_recall"]),
                        "selected_reached_only_by_walk": bool(rlog["selected_reached_only_by_walk"]),
                        "measurement_cost": lb.meter.meas_spent,
                        "compute_cost": lb.meter.compute_spent,
                        "composite_cost": lb.meter.snapshot()["composite_cost"],
                    })

    summary_by_k: dict[str, Any] = {}
    for k_val in k_values:
        summary_by_k[str(k_val)] = {}
        for lbl, _ in modes:
            sub_r = [r for r in rows if r["K"] == k_val and r["arm"] == lbl]
            walk_r = [r for r in sub_r if r["category"] == "T_WalkBridge"]
            assoc_r = [r for r in sub_r if r["category"] == "T_Assoc"]
            summary_by_k[str(k_val)][lbl] = {
                "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in sub_r), 4),
                "confirmed_recall_rate_all": round(mean(1.0 if r["confirmed_via_recall"] else 0.0 for r in sub_r), 4),
                "confirmed_recall_rate_T_Assoc": round(mean(1.0 if r["confirmed_via_recall"] else 0.0 for r in assoc_r), 4),
                "confirmed_recall_rate_T_WalkBridge": round(mean(1.0 if r["confirmed_via_recall"] else 0.0 for r in walk_r), 4),
                "mean_measurement_cost": round(mean(r["measurement_cost"] for r in sub_r), 3),
                "mean_composite_cost": round(mean(r["composite_cost"] for r in sub_r), 3),
            }
    return {"summary_by_k": summary_by_k, "episodes": rows}


def main() -> None:
    t_start = time.perf_counter()
    EVID_DIR.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="hegel_v17_1_run_") as tmp:
        work = Path(tmp)
        closure_res = run_part1_closure_verification(work / "part1")
        bench_res = run_part2_factorized_walk_benchmark(work / "part2")
        iso_k_res = run_part3_isolated_k_scaling(work / "part3")
        open_rev_res = [
            run_open_revision_chain_experiment(sd, work / "part4" / f"seed_{sd}")
            for sd in OPEN_REV_SEEDS
        ]

    # Aggregate clean summary tables for Part 2
    target_rows = bench_res["target_episodes"]
    all_arms = [lbl for lbl, _ in FACTORIZED_FOREST_MODES] + [
        "O_strong_AllValidatedPriors",
        "O_HExA_CodeSkills",
    ]
    arm_summary: dict[str, Any] = {}
    for arm in all_arms:
        arows = [r for r in target_rows if r["arm"] == arm]
        wrows = [r for r in arows if r["category"] == "T_WalkBridge"]
        asrows = [r for r in arows if r["category"] == "T_Assoc"]
        arm_summary[arm] = {
            "n_targets": len(arows),
            "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in arows), 4),
            "false_confirmations": sum(1 for r in arows if r["false_confirmation"]),
            "confirmed_via_recall_all": f"{sum(1 for r in arows if r['confirmed_via_recall'])}/{len(arows)}",
            "confirmed_via_recall_T_Assoc": f"{sum(1 for r in asrows if r['confirmed_via_recall'])}/{len(asrows)}",
            "confirmed_via_recall_T_WalkBridge": f"{sum(1 for r in wrows if r['confirmed_via_recall'])}/{len(wrows)}",
            "non_direct_walk_wins_T_WalkBridge": f"{sum(1 for r in wrows if r['selected_reached_only_by_walk'])}/{len(wrows)}",
            "mean_measurement_cost_all": round(mean(r["costs"]["measurement_cost"] for r in arows), 3),
            "mean_compute_cost_all": round(mean(r["costs"]["compute_cost"] for r in arows), 3),
            "mean_composite_cost_all": round(mean(r["costs"]["composite_cost"] for r in arows), 3),
            "mean_measurement_cost_T_WalkBridge": round(mean(r["costs"]["measurement_cost"] for r in wrows), 3),
            "mean_composite_cost_T_WalkBridge": round(mean(r["costs"]["composite_cost"] for r in wrows), 3),
            "mean_composite_cost_T_Assoc": round(mean(r["costs"]["composite_cost"] for r in asrows), 3),
        }

    # Write static receipts
    (EVID_DIR / "closure_verification.json").write_text(json.dumps(closure_res, indent=2) + "\n", encoding="utf-8")
    (EVID_DIR / "dev_gate_calibration.json").write_text(
        json.dumps(bench_res["dev_gate_calibration"], indent=2) + "\n", encoding="utf-8"
    )
    (EVID_DIR / "forest_snapshots.json").write_text(
        json.dumps(bench_res["forest_snapshots"], indent=2) + "\n", encoding="utf-8"
    )
    (EVID_DIR / "source_episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in bench_res["source_episodes"]) + "\n", encoding="utf-8"
    )
    (EVID_DIR / "target_episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in target_rows) + "\n", encoding="utf-8"
    )
    (EVID_DIR / "isolated_k_scaling.json").write_text(json.dumps(iso_k_res, indent=2) + "\n", encoding="utf-8")
    (EVID_DIR / "open_revision_fair_results.json").write_text(
        json.dumps(open_rev_res, indent=2) + "\n", encoding="utf-8"
    )

    code_modules = [
        "src/hegel_machine/cognition/wormhole_forest.py",
        "src/hegel_machine/cognition/open_workspace_labs.py",
        "src/hegel_machine/cognition/open_workspace_wormhole.py",
        "src/hegel_machine/cognition/open_python_revision.py",
        "src/hegel_machine/cognition/hipporag_wormhole_index.py",
        "src/hegel_machine/cognition/m3_scale_universe.py",
        "src/hegel_machine/cognition/m3_1_walk_factorization.py",
    ]
    manifest = {
        "version": "V17.1_M3_1_Audit_Closure_Factorization_and_MultiHop_Walk",
        "date": "2026-11-19",
        "elapsed_wall_seconds": round(time.perf_counter() - t_start, 2),
        "module_sha256": {p: _sha256_file(REPO_ROOT / p) for p in code_modules},
        "closure_verification": closure_res,
        "oos_honesty_summary": {
            "total_oos_runs": len(bench_res["oos_episodes"]),
            "scoped_unknown_count": sum(1 for r in bench_res["oos_episodes"] if r["submission_status"] == "scoped_unknown"),
            "false_confirmations": sum(1 for r in bench_res["oos_episodes"] if r["false_confirmation"]),
        },
        "factorized_arm_summary": arm_summary,
        "isolated_k_scaling_summary": iso_k_res["summary_by_k"],
    }
    (EVID_DIR / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
