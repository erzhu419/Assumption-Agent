"""Pytest Suite for Hegel V18.3: Complex Multi-Wave Continual Tasks & Offline-to-Online Amortization."""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares

from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    DecisionProvenanceLedger,
    _build_arm_system_and_initial_brief,
    _build_live_k15_forest_and_archive,
)
from hegel_machine.cognition.m5_temporal_multiscale_and_base_subtree import _rmse
from hegel_machine.cognition.m5_v18_3_continual_and_complex_live import (
    apply_online_wave_consolidation,
    build_v18_3_complex_continual_waves,
    compute_offline_to_online_amortization_ledger,
)


def test_cx_complex_waves_disjointness_and_multiround_sublation_structure() -> None:
    waves = build_v18_3_complex_continual_waves()
    assert set(waves.keys()) == {"wave_1", "wave_2", "wave_3"}
    cx01, cx02 = waves["wave_1"]
    cx03, cx04 = waves["wave_2"]
    cx05, cx06 = waves["wave_3"]
    assert cx06.is_out_of_scope is True

    # Verify CX02 smooth siphon achieves ~0 RMSE on pilot (s_grad <= 1.25) yet REFUTES on fresh confirm (s_grad up to 2.15),
    # while adding `high_sgrad_regime_probe` identifies the 5-param threshold-avalanche law with confirm_rmse < 1e-4!
    u_p = np.array([r["u_now"] for r in cx02.pilot_rows])
    s_p = np.array([r["s_grad"] for r in cx02.pilot_rows])
    y_p = np.array([r["y"] for r in cx02.pilot_rows])
    smooth_fit = least_squares(
        lambda p: p[0] * (1.0 - p[1] * s_p / (1.0 + s_p)) * (u_p / (1.2 + u_p)) + p[2] * np.sqrt(u_p) - y_p,
        [6.5, 0.70, 0.3],
    )
    assert float(np.sqrt(np.mean(smooth_fit.fun ** 2))) < 1e-5
    ps = smooth_fit.x
    smooth_conf_preds = [
        ps[0] * (1.0 - ps[1] * r["s_grad"] / (1.0 + r["s_grad"])) * (r["u_now"] / (1.2 + r["u_now"])) + ps[2] * math.sqrt(r["u_now"])
        for r in cx02.confirm_rows
    ]
    assert _rmse(smooth_conf_preds, [r["y"] for r in cx02.confirm_rows]) > 1.0  # Strongly refuted!

    aug_rows = [*cx02.pilot_rows, *cx02.probe_pool["high_sgrad_regime_probe"]["rows"]]
    ua = np.array([r["u_now"] for r in aug_rows])
    sa = np.array([r["s_grad"] for r in aug_rows])
    ya = np.array([r["y"] for r in aug_rows])
    sub_fit = least_squares(
        lambda p: p[0] * (1.0 - p[1] * sa / (1.0 + sa)) * (ua / (1.2 + ua)) + p[2] * np.sqrt(ua) - p[3] * np.maximum(0.0, sa - p[4]) * ua - ya,
        [6.8, 0.70, 0.35, 1.2, 1.4],
        bounds=([1.0, 0.05, -2.0, 0.0, 0.8], [15.0, 0.95, 2.0, 5.0, 2.0]),
    )
    pa = sub_fit.x
    sub_conf_preds = [
        pa[0] * (1.0 - pa[1] * r["s_grad"] / (1.0 + r["s_grad"])) * (r["u_now"] / (1.2 + r["u_now"]))
        + pa[2] * math.sqrt(r["u_now"])
        - pa[3] * max(0.0, r["s_grad"] - pa[4]) * r["u_now"]
        for r in cx02.confirm_rows
    ]
    assert _rmse(sub_conf_preds, [r["y"] for r in cx02.confirm_rows]) < 1e-4
    assert len(cx01.pilot_rows) == 5 and len(cx03.pilot_rows) == 5 and cx04.tau_scale == 0.005
    assert max(r["s_grad"] for r in cx05.pilot_rows) == 1.38


def test_online_wave_consolidation_updates_gray_zone_and_subtree_for_wave3() -> None:
    forest, pb, arch = _build_live_k15_forest_and_archive()
    assert len(forest.nodes) == 15
    cons = apply_online_wave_consolidation(
        forest=forest,
        node_archive=arch,
        wave_receipts=[
            {"task_id": "CX01_SpatioTemporal_SiphonMemory_5P", "status": "confirmed_ok", "strict_pass": True},
            {"task_id": "CX02_CriticalThreshold_AvalancheSublation", "status": "confirmed_ok", "strict_pass": True},
        ],
    )
    assert cons["new_forest_node_count"] == 17
    assert cons["online_writeback_composite_cost"] == 3.6

    # On Wave-3 `CX05` (`pilot max s_grad = 1.38` inside `(1.25, 1.55)` gray zone!), verify smooth siphon is gray-zone blocked!
    waves = build_v18_3_complex_continual_waves()
    cx05 = waves["wave_3"][0]
    led = DecisionProvenanceLedger(run_id="t", task_id=cx05.task_id, arm="Hegel_M5_LiveGPT", model="gpt-6-luna")
    d0 = led.record_decision(
        decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[],
        chosen_option={}, rejection_reasons={}, rationale_summary="init",
    )
    _, brief, _ = _build_arm_system_and_initial_brief(
        cx05, "Hegel_M5_LiveGPT", init_rows=cx05.pilot_rows, forest=forest, priors_bundle=pb,
        node_archive=arch, hippo_idx=HippoRAGWormholeIndex(forest), flat_idx=FlatEqualMetadataIndex(forest),
        ledger=led, parent_id=d0,
    )
    assert any("gray_zone" in str(v) for v in brief["gray_zone_and_scope_rejections"].values())
    assert "GRAY-ZONE GATE BLOCKED SMOOTH SIPHON" in brief["shared_base_subtree_guidance"]["branch_rule"]


def test_offline_to_online_zero_gpu_amortization_breakeven_calculation() -> None:
    hegel_rows = [{"task_id": f"T{i}", "composite_cost": 14.0} for i in range(14)]
    strong_rows = [{"task_id": f"T{i}", "composite_cost": 20.0} for i in range(14)]
    am = compute_offline_to_online_amortization_ledger(
        offline_seed_formation_cost_C=27.608,
        offline_cpu_wall_ms=18.5,
        online_writeback_cost_C=3.6,
        hegel_ordered_task_costs=hegel_rows,
        baseline_ordered_task_costs={"vs_Strong": strong_rows},
    )
    assert am["hardware_profile"]["gpu_required_for_forest_pretrain"] is False
    comp = am["comparisons"]["vs_Strong"]
    # Saving 6.0 C/task recoups 27.608 C at task N* = 5 (5 * 6 = 30 > 27.608)!
    assert comp["breakeven_task_N_seed_amortized"] == 5
    assert comp["net_cold_lifecycle_savings_C"] > 50.0
