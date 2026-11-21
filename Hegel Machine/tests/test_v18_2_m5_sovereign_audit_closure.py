"""Pytest Suite for Hegel V18.2: Sovereign `Hegel_M5_Audit` Closure, De-Confounded Diagnostics & Live Challenge."""

from __future__ import annotations

from pathlib import Path
import random

from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    run_traced_deterministic_m5_episode,
    verify_decision_provenance_chain,
)
from hegel_machine.cognition.m5_temporal_multiscale_and_base_subtree import _build_seed_forest_and_priors
from hegel_machine.cognition.m5_v18_2_sovereign_closure import (
    build_v18_2_live_challenge_tasks,
    run_sovereign_audit_counterexample_closure,
    run_sovereign_twohop_and_narrow_stress_suite,
)


def test_all_hegel_m5_audit_counterexamples_closed_preserving_innovations() -> None:
    repo_hegel_root = Path(__file__).resolve().parents[1]
    res = run_sovereign_audit_counterexample_closure(repo_hegel_root=repo_hegel_root)
    assert res["all_audit_counterexamples_closed"] is True, res
    # Verify progressive Layer-1 (`only_c1`) innovation is preserved while warm-start is gated to Layer-2 (`C1 AND C2`)
    f07 = res["checks"]["F07_joint_binding_and_progressive_layers"]
    assert f07["partial_c1_layer1_preserved"] == "layer1_partial_candidate_only"
    assert f07["partial_c1_warm_start_blocked"] is True
    assert f07["full_c1_and_c2_layer2_warm_authorized"] is True
    # Verify legacy v17_7 W4 gray-zone b=1.80 blocked without touching historical files
    f08 = res["checks"]["F08_legacy_v17_7_gray_zone_containment"]
    assert f08["v17_7_snapshot_semantic_mismatches"] == 0
    assert f08["b_1_80_gray_zone_status"] == "scope_unauthorized"


def test_twohop_wormhole_and_narrow_pilot_sovereign_stress_advantage() -> None:
    stress = run_sovereign_twohop_and_narrow_stress_suite()
    sums = stress["summaries"]
    hegel_ppr = sums["Hegel_M5_Forest_Subtree_PPR"]
    eq_compiled_flat = sums["EqualContent_Flat_CompiledSubtrees_BM25"]
    eq_flat = sums["EqualAuth_Dynamic_Flat_BM25"]
    strong_eq_unit = sums["Strong_Dynamic_EqualUnitPrice_1_5M"]

    assert hegel_ppr["strict_pass"] == "8/8"
    assert hegel_ppr["false_confirmations"] == 0
    # Hegel PPR strictly outperforms even EqualContent_Flat_CompiledSubtrees_BM25 and Strong_Dynamic_EqualUnitPrice_1_5M!
    assert hegel_ppr["measurement_cost"] < eq_compiled_flat["measurement_cost"]
    assert hegel_ppr["composite_cost"] < eq_compiled_flat["composite_cost"]
    assert hegel_ppr["composite_cost"] < eq_flat["composite_cost"]
    assert hegel_ppr["composite_cost"] < strong_eq_unit["composite_cost"]


def test_l07_and_l08_raw_irregular_prefix_without_mem_conv_traced() -> None:
    forest, pb = _build_seed_forest_and_priors()
    h_idx = HippoRAGWormholeIndex(forest)
    f_idx = FlatEqualMetadataIndex(forest)
    l07, l08 = build_v18_2_live_challenge_tasks()

    # L08 strictly has mem_conv == None on every row
    assert all(r["mem_conv"] is None and len(r["raw_prefix"]) >= 8 for r in l08.pilot_rows)
    assert all(r["raw_prefix"][0]["t"] >= 100.0 for r in l08.pilot_rows)

    for ds in (l07, l08):
        ep, chain = run_traced_deterministic_m5_episode(
            ds, "Hegel_M5_Full_TemporalSubtree_Forest",
            forest=forest, priors_bundle=pb, hippo_idx=h_idx, flat_idx=f_idx, rng=random.Random(2026),
        )
        assert ep["strict_pass"] is True, ep
        assert ep["false_confirmation"] is False
        assert verify_decision_provenance_chain(chain)["valid"] is True
