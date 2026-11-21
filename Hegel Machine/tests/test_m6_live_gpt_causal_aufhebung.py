"""Pre-flight Unit Tests for M6 vs M5 Live GPT Causal Aufhebung Benchmark (`exp/m6-causal-wormhole-sandbox`)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from hegel_machine.cognition.hipporag_wormhole_index import (
    FlatEqualMetadataIndex,
    HippoRAGWormholeIndex,
)
from hegel_machine.cognition.m5_live_gpt_and_provenance import DecisionProvenanceLedger
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import (
    apply_post_wave1_consolidation,
    build_m6_arm_brief,
    build_m6_causal_24_waves,
    build_m6_k40_causal_forest_and_archive,
    run_m6_live_gpt_causal_aufhebung_study,
)


def _row_sig(r: dict) -> str:
    return json.dumps({k: v for k, v in r.items() if k != "y"}, sort_keys=True)


def test_m6_k40_forest_and_24_disjoint_causal_tasks() -> None:
    forest, arch, whs = build_m6_k40_causal_forest_and_archive()
    assert len(forest.nodes) == 40
    assert len(arch) == 40
    assert len(whs) == 8

    waves = build_m6_causal_24_waves(20261121)
    assert len(waves["wave_1"]) == 12
    assert len(waves["wave_2"]) == 12

    for w in ("wave_1", "wave_2"):
        for ds in waves[w]:
            pk = {_row_sig(r) for r in ds.pilot_rows}
            ck = {_row_sig(r) for r in ds.confirm_rows}
            hk = {_row_sig(r) for r in ds.heldout_rows}
            assert pk.isdisjoint(ck)
            assert pk.isdisjoint(hk)
            assert ck.isdisjoint(hk)
            for r in (*ds.pilot_rows, *ds.confirm_rows, *ds.heldout_rows):
                assert math.isfinite(float(r["y"]))


def test_m6_vs_m5_post_wave1_consolidation_and_recursive_pullback(tmp_path: Path) -> None:
    f_m5, arch_m5, _ = build_m6_k40_causal_forest_and_archive()
    f_m6, arch_m6, _ = build_m6_k40_causal_forest_and_archive()

    w1_ok = [
        {"subfamily": "C01_ConfounderFork_RetroactiveImmunity", "status": "confirmed_ok", "strict_pass": True},
        {"subfamily": "C02_MediatorDecomposition_WormholeReAnchor", "status": "confirmed_ok", "strict_pass": True},
        {"subfamily": "C03_ColliderSelection_BerksonDebiasing", "status": "confirmed_ok", "strict_pass": True},
    ]

    apply_post_wave1_consolidation(arm="Hegel_M5_GrowthOnly_LiveGPT", forest=f_m5, node_archive=arch_m5, wave1_receipts=w1_ok)
    cons_m6 = apply_post_wave1_consolidation(arm="Hegel_M6_Aufhebung_LiveGPT", forest=f_m6, node_archive=arch_m6, wave1_receipts=w1_ok)

    # 1. M5 only revised local Tree_A (`node_C01_TreeA_prime_electro` & `node_C01_TreeF_doubleprime_catalysis` STILL say `x_ind / 1.5`!)
    assert "z_2" not in arch_m5["node_C01_TreeA_prime_electro"]["verified_mechanism_note"]
    assert "z_2" not in arch_m5["node_C01_TreeF_doubleprime_catalysis"]["verified_mechanism_note"]

    # 2. M6 retroactively re-interpreted BOTH 1-hop `Tree_A_prime` AND 2-hop `Tree_F_double_prime` (`z_2` confounder),
    #    while Bayes-Ball `d-separation` kept `node_C05_TreeG_orth_direct` 100% untouched!
    assert "z_2" in arch_m6["node_C01_TreeA_prime_electro"]["verified_mechanism_note"]
    assert "z_2" in arch_m6["node_C01_TreeF_doubleprime_catalysis"]["verified_mechanism_note"]
    assert arch_m6["node_C05_TreeG_orth_direct"]["epistemic_rung"] == "L2_INVARIANT_CAUSAL"
    assert "x_ind / (1.1 + x_ind)" in arch_m6["node_C05_TreeG_orth_direct"]["verified_mechanism_note"]
    assert len(cons_m6["events"]) == 4

    # 3. Verify C04 Wormhole-First Recursive Pullback expands Depth 2 & 3 on M6, NOT on M5!
    waves = build_m6_causal_24_waves(20261121)
    ds_c04 = next(d for d in waves["wave_1"] if d.subfamily == "C04_WormholeFirst_MultiDepth_RootPullback")
    hippo = HippoRAGWormholeIndex(f_m6)
    flat = FlatEqualMetadataIndex(f_m6)

    led5 = DecisionProvenanceLedger(run_id="r5", task_id=ds_c04.task_id, arm="Hegel_M5_GrowthOnly_LiveGPT", model="m", provider_backend="mock")
    p5 = led5.record_decision(decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[], chosen_option={}, rejection_reasons={}, rationale_summary="i")
    _, b5, _ = build_m6_arm_brief(ds_c04, "Hegel_M5_GrowthOnly_LiveGPT", forest=f_m5, node_archive=arch_m5, hippo_idx=hippo, flat_idx=flat, ledger=led5, parent_id=p5)
    assert "m6_recursive_subassumption_pullback" not in b5

    led6 = DecisionProvenanceLedger(run_id="r6", task_id=ds_c04.task_id, arm="Hegel_M6_Aufhebung_LiveGPT", model="m", provider_backend="mock")
    p6 = led6.record_decision(decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[], chosen_option={}, rejection_reasons={}, rationale_summary="i")
    _, b6, _ = build_m6_arm_brief(ds_c04, "Hegel_M6_Aufhebung_LiveGPT", forest=f_m6, node_archive=arch_m6, hippo_idx=hippo, flat_idx=flat, ledger=led6, parent_id=p6)
    assert b6["m6_recursive_subassumption_pullback"] is not None
    assert "node_C04_Depth2_IndoorSpaceSigmoid" in b6["m6_recursive_subassumption_pullback"]["wormhole_reused_subassumptions_depth_2_3"]
    assert "node_C04_Depth3_BiomechImpactPenalty" in b6["m6_recursive_subassumption_pullback"]["wormhole_reused_subassumptions_depth_2_3"]

    dry = run_m6_live_gpt_causal_aufhebung_study(tmp_path / "dry_m6", run_live_gpt=False)
    assert dry["total_tasks"] == 24
