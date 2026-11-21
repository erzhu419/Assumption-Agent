"""Unit & Theorem Verification Tests for M6 Phase-0 Causal-Wormhole Sandbox (`exp/m6-causal-wormhole-sandbox`)."""

from __future__ import annotations

from pathlib import Path

from hegel_machine.cognition.m6_causal_wormhole_sandbox import (
    is_d_separated_dag,
    run_canonical_motifs_r1_r2_r3_suite,
    run_m6_phase0_complete_study,
)


def test_bayes_ball_fork_chain_and_collider_d_separation() -> None:
    # 1. Confounder fork `X <- Z -> Y`: d-connected unconditionally, d-separated given `Z`!
    fork = [("Z", "X"), ("Z", "Y")]
    assert not is_d_separated_dag(fork, {"X"}, {"Y"}, set())
    assert is_d_separated_dag(fork, {"X"}, {"Y"}, {"Z"})

    # 2. Collider `X -> C <- W`: d-separated unconditionally, d-connected when conditioning on `C` or child `D` of `C`!
    col = [("X", "C"), ("W", "C"), ("C", "D")]
    assert is_d_separated_dag(col, {"X"}, {"W"}, set())
    assert not is_d_separated_dag(col, {"X"}, {"W"}, {"C"})
    assert not is_d_separated_dag(col, {"X"}, {"W"}, {"D"})


def test_m6_phase0_motifs_multihop_radius_and_recursive_pullback(tmp_path: Path) -> None:
    manifest = run_m6_phase0_complete_study(tmp_path / "m6_ev")

    # Suite 1: R1 Fork, R2 Mediator, R3 Collider
    r1 = manifest["suite_1_canonical_causal_motifs"]["R1_ConfounderFork"]
    assert r1["d_separated_given_z"] is True
    assert r1["retroactive_projection_certificate"]["certified"] is True
    assert r1["m6_wave2_sibling_interventional_rmse"] < 0.25 * r1["m5_wave2_sibling_interventional_rmse"]

    r2 = manifest["suite_1_canonical_causal_motifs"]["R2_MediatorWormholeReAnchoring"]
    assert r2["m6_reanchored_transfer_rmse"] == 0.0
    assert r2["m5_unreanchored_transfer_rmse"] > 1.5

    # Suite 2: Multi-hop `A <-> A' <-> F''` (`K=80` nodes, `125` forests across densities `0.10..0.70`)
    ov = manifest["suite_2_multihop_wormhole_propagation_k80"]["overall_summary"]
    assert ov["B0_LocalPatch_0Hop"]["downstream_causal_recall"] == 0.0
    assert 0.45 <= ov["B1_Fixed_1Hop_Blanket"]["downstream_causal_recall"] < 0.80  # Misses multi-hop T02/T03 unless hit accidentally!
    assert ov["B2_Unscreened_Transitive_Closure"]["outside_blanket_false_touch_rate"] > 0.60  # Percolation explosion!

    m6 = ov["M6_dSepset_ATMS_RetroactiveAufhebung"]
    assert m6["downstream_causal_recall"] == 1.0  # Catches 1-hop, 2-hop, AND 3-hop causal chain!
    assert m6["outside_blanket_false_touch_rate"] == 0.0  # Zero false alarms on collider-blocked & invariant trees!
    assert m6["outside_blanket_regression_rate"] == 0.0
    assert m6["touched_nodes_out_of_80"] == 25.0  # Strictly O(1) = 5 trees * 5 nodes regardless of background wormhole density!
    assert m6["wave2_zeroshot_pass_rate"] == 1.0

    # Suite 3: Wormhole-first multi-depth recursive sub-assumption (`BrainPower <- Exercise <- ... -> Elliptical => BrainPower`)
    s3 = manifest["suite_3_wormhole_first_recursive_subassumption"]["arms"]
    assert s3["SingleLayer_Growth_Only"]["root_goal_satisfied"] is False
    assert s3["Unbounded_NoRootPullback"]["root_goal_satisfied"] is False
    assert s3["M6_WormholeFirst_RecursivePullback"]["root_goal_satisfied"] is True
    assert s3["M6_WormholeFirst_RecursivePullback"]["cost_reduction_vs_no_wormhole_pct"] > 75.0
