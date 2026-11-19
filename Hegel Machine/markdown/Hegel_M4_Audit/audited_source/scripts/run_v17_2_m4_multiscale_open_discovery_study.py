"""Pre-registered Hegel V17.2 (`M4`) Study:
Open Compositional AST Discovery, Active Multi-Scale Renormalization & K=64 Continual Forest Lifecycle.

Writes full audit receipts, persisted authored Python artifacts, per-cluster frozen gates,
multi-scale closure certificates, 7-arm target ledgers (`504` target runs + `8` OOS runs),
and 4-stage `K = 16 -> 32 -> 48 -> 64` continual compression/forgetting curves to:
  `docs/evidence/v17_2_m4_multiscale_open_discovery_20261119/`
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m4_compositional_discovery import run_m4_compositional_discovery_suite
from hegel_machine.cognition.m4_multiscale_renormalization import (
    build_m4_cluster_targets,
    build_m4_multiscale_forest,
    evaluate_m4_target_under_arm,
    run_m4_continual_lifecycle_stream,
)
from hegel_machine.cognition.wormhole_forest import TwoStageWormholeGate


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v17_2_m4_multiscale_open_discovery_20261119"

TRACKED_MODULES = (
    "src/hegel_machine/cognition/wormhole_forest.py",
    "src/hegel_machine/cognition/hipporag_wormhole_index.py",
    "src/hegel_machine/cognition/open_workspace_labs.py",
    "src/hegel_machine/cognition/open_workspace_wormhole.py",
    "src/hegel_machine/cognition/m4_compositional_discovery.py",
    "src/hegel_machine/cognition/m4_multiscale_renormalization.py",
)

M4_ARMS = (
    "Hegel_M4_MultiScale_HippoWalk",
    "HippoPPR_Alpha0_MultiScalePool",
    "HippoPPR_ScaleBlind_Walk",
    "Flat_BM25_WormholePool_EqualContent",
    "Forest_VerticalOnly_NoWormholes",
    "Continuing_AllHistory_CodeAuthor",
    "MicroOnly_AlwaysFine_Forest",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.perf_counter()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    authored_archive_dir = EVIDENCE_DIR / "persisted_open_author_artifacts"
    if authored_archive_dir.exists():
        shutil.rmtree(authored_archive_dir)
    authored_archive_dir.mkdir(parents=True, exist_ok=True)

    module_hashes = {rel: _sha256_file(REPO_ROOT / rel) for rel in TRACKED_MODULES}

    with TemporaryDirectory(prefix="hegel-v17-2-m4-") as tmp_str:
        tmp_root = Path(tmp_str)

        # ====================================================================
        # Pillar 1: Open Compositional AST Synthesis & Counterfactual Witnesses
        # ====================================================================
        comp_work = tmp_root / "pillar1_compositional"
        comp_suite = run_m4_compositional_discovery_suite(comp_work)

        # Persist actual authored Python models, method JSONs & training prefixes into evidence!
        for sub_name in (
            "S1_monod",
            "W_haldane_shunt",
            "W_variational_cap",
            "W_pure_shunt",
            "W_underdamped_ringing",
            "W_fast_mode_excess",
            "target_Hegel_Intact_Artifact",
            "target_Hegel_Deleted_Artifact",
            "target_Hegel_Restored_Artifact",
            "target_Ordinary_AllHistory",
        ):
            src_sub = comp_work / sub_name
            dst_sub = authored_archive_dir / sub_name
            dst_sub.mkdir(parents=True, exist_ok=True)
            for fname in ("authored_model.py", "authored_method.json", "observations.json", "confirmed_training_observations.json"):
                if (src_sub / fname).is_file():
                    shutil.copyfile(src_sub / fname, dst_sub / fname)

        # ====================================================================
        # DEV Calibration of Frozen TwoStageWormholeGate (`cluster_idx = 100`)
        # ====================================================================
        dev_forest, dev_src_meta = build_m4_multiscale_forest(cluster_seed=8800, extra_distractor_count=12)
        dev_targets, _ = build_m4_cluster_targets(cluster_idx=100)
        dev_idx = HippoRAGWormholeIndex(dev_forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        calibrated_gate = TwoStageWormholeGate(margin=0.25)
        dev_composite_total = float(dev_src_meta["source_formation_and_bridge_cost"]) + dev_idx.index_build_compute_cost * 0.05

        for dt in dev_targets[:8]:
            tr_res = evaluate_m4_target_under_arm(
                dt,
                arm_name="Hegel_M4_MultiScale_HippoWalk",
                forest=dev_forest,
                index=dev_idx,
                stage1_gate=None,
                work_dir=tmp_root / f"dev_tr_{dt.target_id}",
            )
            dir_res = evaluate_m4_target_under_arm(
                dt,
                arm_name="Continuing_AllHistory_CodeAuthor",
                forest=dev_forest,
                index=None,
                stage1_gate=None,
                work_dir=tmp_root / f"dev_dir_{dt.target_id}",
            )
            sig_key = f"arity_{len(dt.lab.feature_names)}_{tr_res['selected_pilot_channel']}"
            calibrated_gate.observe_paired_dev_rollout(
                sig_key,
                transfer_remaining_cost=tr_res["cost"]["composite_cost"],
                direct_remaining_cost=dir_res["cost"]["composite_cost"],
                transfer_quality_ok=tr_res["strict_pass"],
                direct_quality_ok=dir_res["strict_pass"],
            )
            dev_composite_total += tr_res["cost"]["composite_cost"] + dir_res["cost"]["composite_cost"]

        # ====================================================================
        # Pillar 2: 4 Frozen Multi-Scale Eval Clusters × 18 Targets × 7 Arms
        # ====================================================================
        all_target_rows: list[dict[str, Any]] = []
        oos_rows: list[dict[str, Any]] = []
        cluster_metadata: list[dict[str, Any]] = []

        for c_idx in range(4):
            forest, src_meta = build_m4_multiscale_forest(cluster_seed=5100 + 97 * c_idx, extra_distractor_count=12)
            targets, oos_targets = build_m4_cluster_targets(cluster_idx=c_idx)

            idx_walk = HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
            idx_alpha0 = HippoRAGWormholeIndex(forest, damping_alpha=0.0, wormhole_equivalence_pooling=True, scale_aware=True)
            idx_blind = HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=False)
            idx_flat = FlatEqualMetadataIndex(forest, mode="bm25", wormhole_equivalence_pooling=True)
            idx_vert = HippoRAGWormholeIndex(
                forest, allow_horizontal_wormholes=False, wormhole_equivalence_pooling=False, damping_alpha=0.65, scale_aware=True
            )

            arm_index_map = {
                "Hegel_M4_MultiScale_HippoWalk": idx_walk,
                "HippoPPR_Alpha0_MultiScalePool": idx_alpha0,
                "HippoPPR_ScaleBlind_Walk": idx_blind,
                "Flat_BM25_WormholePool_EqualContent": idx_flat,
                "Forest_VerticalOnly_NoWormholes": idx_vert,
                "Continuing_AllHistory_CodeAuthor": None,
                "MicroOnly_AlwaysFine_Forest": idx_walk,
            }

            cluster_metadata.append({
                "cluster_idx": c_idx,
                "source_meta": src_meta,
                "edge_counts": idx_walk.edge_counts,
                "index_build_compute_cost": idx_walk.index_build_compute_cost,
            })

            for arm_name in M4_ARMS:
                gate_clone = TwoStageWormholeGate(margin=0.25)
                gate_clone.paired_dev_records = {k: list(v) for k, v in calibrated_gate.paired_dev_records.items()}
                for t in targets:
                    res = evaluate_m4_target_under_arm(
                        t,
                        arm_name=arm_name,
                        forest=forest,
                        index=arm_index_map[arm_name],
                        stage1_gate=(gate_clone if arm_name != "Continuing_AllHistory_CodeAuthor" else None),
                        work_dir=tmp_root / f"c{c_idx}_{arm_name}_{t.target_id}",
                    )
                    res["cluster_idx"] = c_idx
                    all_target_rows.append(res)

            for ot in oos_targets:
                ores = evaluate_m4_target_under_arm(
                    ot,
                    arm_name="Hegel_M4_MultiScale_HippoWalk",
                    forest=forest,
                    index=idx_walk,
                    stage1_gate=None,
                    work_dir=tmp_root / f"c{c_idx}_oos_{ot.target_id}",
                )
                ores["cluster_idx"] = c_idx
                oos_rows.append(ores)

        # ====================================================================
        # Pillar 3: Continual K = 16 -> 32 -> 48 -> 64 Self-Pruning Lifecycle
        # ====================================================================
        continual_report = run_m4_continual_lifecycle_stream(tmp_root / "pillar3_continual_k64")

    # Aggregate 7-arm Multi-Scale Summary
    arm_summaries: dict[str, Any] = {}
    for arm_name in M4_ARMS:
        rows = [r for r in all_target_rows if r["arm"] == arm_name]
        n = len(rows)
        adiab = [r for r in rows if r["category"] == "T_Adiabatic_Coarse"]
        brk = [r for r in rows if r["category"] == "T_Closure_Breakdown"]
        wlk = [r for r in rows if r["category"] == "T_MultiScale_Walk"]
        prm = [r for r in rows if r["category"] == "T_DualSensor_Permuted"]

        arm_summaries[arm_name] = {
            "n_targets": n,
            "strict_pass_rate": round(sum(1 for r in rows if r["strict_pass"]) / n, 4),
            "false_confirmations": sum(1 for r in rows if r["false_confirmation"]),
            "confirmed_via_recall_all": f"{sum(1 for r in rows if r['confirmed_via_recall'])}/{n}",
            "non_direct_walk_wins_T_MultiScale_Walk": f"{sum(1 for r in wlk if r['non_direct_walk_win'])}/{len(wlk)}",
            "escalated_to_micro_T_Closure_Breakdown": f"{sum(1 for r in brk if r['escalated_to_micro'])}/{len(brk)}",
            "mean_measurement_cost_all": round(sum(r["cost"]["measurement_cost"] for r in rows) / n, 3),
            "mean_compute_cost_all": round(sum(r["cost"]["compute_cost"] for r in rows) / n, 3),
            "mean_composite_cost_all": round(sum(r["cost"]["composite_cost"] for r in rows) / n, 3),
            "mean_measurement_cost_T_Adiabatic_Coarse": round(sum(r["cost"]["measurement_cost"] for r in adiab) / len(adiab), 3),
            "mean_composite_cost_T_Closure_Breakdown": round(sum(r["cost"]["composite_cost"] for r in brk) / len(brk), 3),
            "mean_composite_cost_T_MultiScale_Walk": round(sum(r["cost"]["composite_cost"] for r in wlk) / len(wlk), 3),
            "mean_composite_cost_T_DualSensor_Permuted": round(sum(r["cost"]["composite_cost"] for r in prm) / len(prm), 3),
        }

    elapsed_wall = round(time.perf_counter() - started, 2)
    summary_payload = {
        "version": "V17.2_M4_Open_Compositional_Discovery_MultiScale_Renormalization_and_K64_Lifecycle",
        "date": "2026-11-19",
        "elapsed_wall_seconds": elapsed_wall,
        "module_sha256": module_hashes,
        "dev_gate_calibration": {
            "dev_total_composite_cost": round(dev_composite_total, 3),
            "paired_dev_keys": {k: len(v) for k, v in calibrated_gate.paired_dev_records.items()},
        },
        "pillar1_open_compositional_discovery": {
            "counterfactual_summary": comp_suite["counterfactual_summary"],
            "witness_ast_digests": {
                k: {
                    "family_hint": v["authored_artifact"]["operator_spec"]["family_hint"],
                    "ast_skeleton_digest": v["ast_info"]["ast_skeleton_digest"],
                    "skeleton_status": v["ast_info"]["skeleton_status"],
                    "diagnostic_report": v["authored_artifact"]["diagnostic_report"],
                    "confirm_rmse": v["confirm_rmse"],
                    "heldout_rmse": v["heldout_rmse"],
                    "strict_pass": v["strict_pass"],
                    "backward_s1_preserved": v.get("backward_s1_preserved"),
                }
                for k, v in comp_suite["counterfactual_witnesses"].items()
            },
            "artifact_causality_and_transfer": {
                k: (
                    {
                        "strict_pass": v["strict_pass"],
                        "ran_diagnostic": v["ran_diagnostic"],
                        "evaluated_candidate_count": v["evaluated_candidate_count"],
                        "measurement_cost": v["cost"]["measurement_cost"],
                        "compute_cost": v["cost"]["compute_cost"],
                        "composite_cost": v["cost"]["composite_cost"],
                    }
                    if isinstance(v, dict) and "cost" in v
                    else v
                )
                for k, v in comp_suite["artifact_causality_and_transfer"].items()
            },
        },
        "pillar2_multiscale_7arm_summary": arm_summaries,
        "pillar2_oos_honesty_summary": {
            "total_oos_runs": len(oos_rows),
            "scoped_unknown_count": sum(1 for r in oos_rows if r["status"] == "scoped_unknown"),
            "false_confirmations": sum(1 for r in oos_rows if r["false_confirmation"]),
        },
        "pillar2_cluster_graph_metadata": cluster_metadata,
        "pillar3_continual_k64_lifecycle": continual_report,
    }

    (EVIDENCE_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    (EVIDENCE_DIR / "multiscale_target_runs.json").write_text(json.dumps(all_target_rows, indent=2), encoding="utf-8")
    (EVIDENCE_DIR / "continual_k64_stages.json").write_text(json.dumps(continual_report, indent=2), encoding="utf-8")

    with (EVIDENCE_DIR / "multiscale_7arm_table.csv").open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "arm", "n_targets", "strict_pass_rate", "false_confirmations",
            "confirmed_via_recall_all", "walk_wins_T_MultiScale_Walk",
            "escalated_T_Closure_Breakdown", "mean_measurement_cost",
            "mean_compute_cost", "mean_composite_cost",
            "mean_meas_Adiabatic", "mean_comp_Breakdown", "mean_comp_Walk",
        ])
        for arm_name, m in arm_summaries.items():
            writer.writerow([
                arm_name, m["n_targets"], m["strict_pass_rate"], m["false_confirmations"],
                m["confirmed_via_recall_all"], m["non_direct_walk_wins_T_MultiScale_Walk"],
                m["escalated_to_micro_T_Closure_Breakdown"], m["mean_measurement_cost_all"],
                m["mean_compute_cost_all"], m["mean_composite_cost_all"],
                m["mean_measurement_cost_T_Adiabatic_Coarse"],
                m["mean_composite_cost_T_Closure_Breakdown"],
                m["mean_composite_cost_T_MultiScale_Walk"],
            ])

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
