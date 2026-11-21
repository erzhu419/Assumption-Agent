"""Pre-flight & Mechanical Verification Tests for V18.4 (`Hegel-Discovery-240`, K=45->48, 240 Tasks)."""

from __future__ import annotations

from pathlib import Path
import math

import pytest

from hegel_machine.cognition.hipporag_wormhole_index import (
    FlatEqualMetadataIndex,
    HippoRAGWormholeIndex,
)
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    DecisionProvenanceLedger,
    SubscriptionQuotaPausedError,
    _build_arm_system_and_initial_brief,
    _row_input_key,
    run_single_live_gpt_m5_episode,
)
from hegel_machine.cognition.m5_v18_4_hegel_discovery_240 import (
    IN_SCOPE_12_FAMILIES,
    apply_v18_4_online_wave_consolidation,
    build_hegel_discovery_240_waves,
    build_live_k45_forest_and_archive,
    run_v18_4_hegel_discovery_240_study,
)


def test_v18_4_k45_forest_and_240_disjoint_tasks_integrity() -> None:
    forest, priors_bundle, node_archive = build_live_k45_forest_and_archive()
    assert len(forest.nodes) == 45
    assert len(node_archive) == 45
    assert "sub_2layer_siphon_base" in forest.shared_base_subtrees
    assert priors_bundle["source_formation_cost"] > 0.0

    waves = build_hegel_discovery_240_waves(20261121)
    assert list(waves.keys()) == [f"wave_{i}" for i in range(1, 7)]
    all_tasks = [t for w in waves.values() for t in w]
    assert len(all_tasks) == 240

    seen_ids = {t.task_id for t in all_tasks}
    assert len(seen_ids) == 240

    in_scope = [t for t in all_tasks if not t.is_out_of_scope]
    oos = [t for t in all_tasks if t.is_out_of_scope]
    assert len(in_scope) == 216
    assert len(oos) == 24

    for fam in IN_SCOPE_12_FAMILIES:
        fam_tasks = [t for t in in_scope if t.subfamily == fam]
        assert len(fam_tasks) == 18, f"Expected 18 tasks for {fam}, got {len(fam_tasks)}"

    for ds in all_tasks:
        p_keys = {_row_input_key(r) for r in ds.pilot_rows}
        c_keys = {_row_input_key(r) for r in ds.confirm_rows}
        h_keys = {_row_input_key(r) for r in ds.heldout_rows}
        assert p_keys.isdisjoint(c_keys), f"Pilot/confirm overlap in {ds.task_id}"
        assert p_keys.isdisjoint(h_keys), f"Pilot/heldout overlap in {ds.task_id}"
        assert c_keys.isdisjoint(h_keys), f"Confirm/heldout overlap in {ds.task_id}"
        for r in (*ds.pilot_rows, *ds.confirm_rows, *ds.heldout_rows):
            assert math.isfinite(float(r["y"]))


def test_v18_4_k45_wormhole_bridge_vs_flat_bm25_and_f09_oos_briefs() -> None:
    forest, priors_bundle, node_archive = build_live_k45_forest_and_archive()
    hippo = HippoRAGWormholeIndex(forest)
    flat = FlatEqualMetadataIndex(forest)
    waves = build_hegel_discovery_240_waves(20261121)

    # 1. F08 2-hop cross-disciplinary wormhole bridge: HippoRAG recalls `node_C_source_inland_hub` in Top-3; Flat BM25 misses!
    tv = {"goal_kind": "trajectory_and_horizon_prediction", "tolerance": 0.045}
    f08_tasks = [t for t in waves["wave_1"] if t.subfamily == "F08_TwoHop_WormholeBridge_Transfer"]
    assert len(f08_tasks) == 3
    for ds_f08 in f08_tasks:
        h_ids = [c["node_id"] for c in hippo.retrieve(ds_f08.probe_signatures, task_view=tv, top_k=3)["candidates"]]
        b_ids = [c["node_id"] for c in flat.retrieve(ds_f08.probe_signatures, task_view=tv, top_k=3)["candidates"]]
        assert "node_C_source_inland_hub" in h_ids, f"HippoRAG missed node_C on {ds_f08.task_id}: {h_ids}"
        assert "node_C_source_inland_hub" not in b_ids, f"Flat BM25 unexpectedly recalled node_C on {ds_f08.task_id}: {b_ids}"

        led = DecisionProvenanceLedger(run_id="t", task_id=ds_f08.task_id, arm="Hegel_M5_LiveGPT", model="m", provider_backend="mock")
        p0 = led.record_decision(decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[], chosen_option={}, rejection_reasons={}, rationale_summary="init")
        _, brief_f08, _ = _build_arm_system_and_initial_brief(
            ds_f08, "Hegel_M5_LiveGPT", init_rows=ds_f08.pilot_rows, forest=forest,
            priors_bundle=priors_bundle, node_archive=node_archive, hippo_idx=hippo, flat_idx=flat,
            ledger=led, parent_id=p0,
        )
        assert brief_f08["shared_base_subtree_guidance"] is not None
        assert "open_branch_discriminating_probe" in brief_f08["shared_base_subtree_guidance"]["branch_rule"]

    # 2. F09 LSR Hill-2 competitive inhibition: guides Hill-2 (`u_n ** 2`), NOT standard Michaelis siphon!
    ds_f09 = next(t for t in waves["wave_1"] if t.subfamily == "F09_LSR_Transformed_HillInhibition")
    led9 = DecisionProvenanceLedger(run_id="t9", task_id=ds_f09.task_id, arm="Hegel_M5_LiveGPT", model="m", provider_backend="mock")
    p9 = led9.record_decision(decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[], chosen_option={}, rejection_reasons={}, rationale_summary="init")
    _, brief_f09, _ = _build_arm_system_and_initial_brief(
        ds_f09, "Hegel_M5_LiveGPT", init_rows=ds_f09.pilot_rows, forest=forest,
        priors_bundle=priors_bundle, node_archive=node_archive, hippo_idx=hippo, flat_idx=flat,
        ledger=led9, parent_id=p9,
    )
    assert brief_f09["shared_base_subtree_guidance"] is not None
    assert "k_half" in brief_f09["shared_base_subtree_guidance"]["branch_rule"]
    assert "alpha_inh" in brief_f09["shared_base_subtree_guidance"]["branch_rule"]

    # 3. OOS adversarial tasks: zero spurious `shared_base_subtree_guidance` even with lure signatures!
    for ds_oos in [t for t in waves["wave_1"] if t.is_out_of_scope]:
        ledo = DecisionProvenanceLedger(run_id="to", task_id=ds_oos.task_id, arm="Hegel_M5_LiveGPT", model="m", provider_backend="mock")
        po = ledo.record_decision(decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[], chosen_option={}, rejection_reasons={}, rationale_summary="init")
        _, brief_oos, _ = _build_arm_system_and_initial_brief(
            ds_oos, "Hegel_M5_LiveGPT", init_rows=ds_oos.pilot_rows, forest=forest,
            priors_bundle=priors_bundle, node_archive=node_archive, hippo_idx=hippo, flat_idx=flat,
            ledger=ledo, parent_id=po,
        )
        assert brief_oos["shared_base_subtree_guidance"] is None


def test_v18_4_continual_wave_consolidation_and_zero_subcritical_interference() -> None:
    forest, priors_bundle, node_archive = build_live_k45_forest_and_archive()
    waves = build_hegel_discovery_240_waves(20261121)

    cons = apply_v18_4_online_wave_consolidation(
        forest=forest,
        node_archive=node_archive,
        wave_receipts=[
            {"subfamily": "F07_CriticalThreshold_GrayZone_Avalanche", "task_id": "W1R1__F07", "status": "confirmed_ok", "strict_pass": True},
            {"subfamily": "F11_SpatioTemporal_SiphonMemory_5P", "task_id": "W1R1__F11", "status": "confirmed_ok", "strict_pass": True},
            {"subfamily": "F12_SinkCoupled_DualHorizon_5P", "task_id": "W1R1__F12", "status": "confirmed_ok", "strict_pass": True},
        ],
    )
    assert cons["new_forest_node_count"] == 48
    assert cons["online_writeback_composite_cost"] == pytest.approx(5.40)

    hippo = HippoRAGWormholeIndex(forest)
    flat = FlatEqualMetadataIndex(forest)

    # Wave 2 F07 (`s_grad = 2.44` inside `[2.36, 2.68]`): MUST trigger Turn-1 gray-zone warning!
    ds_w2_f07 = next(t for t in waves["wave_2"] if t.subfamily == "F07_CriticalThreshold_GrayZone_Avalanche")
    led7 = DecisionProvenanceLedger(run_id="t7", task_id=ds_w2_f07.task_id, arm="Hegel_M5_LiveGPT", model="m", provider_backend="mock")
    p7 = led7.record_decision(decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[], chosen_option={}, rejection_reasons={}, rationale_summary="init")
    _, brief_w2_f07, _ = _build_arm_system_and_initial_brief(
        ds_w2_f07, "Hegel_M5_LiveGPT", init_rows=ds_w2_f07.pilot_rows, forest=forest,
        priors_bundle=priors_bundle, node_archive=node_archive, hippo_idx=hippo, flat_idx=flat,
        ledger=led7, parent_id=p7,
    )
    assert "GRAY-ZONE GATE BLOCKED SMOOTH SIPHON" in brief_w2_f07["shared_base_subtree_guidance"]["branch_rule"]

    # Wave 2 F05 (`s_grad <= 2.20`): MUST NOT trigger gray-zone false positive!
    ds_w2_f05 = next(t for t in waves["wave_2"] if t.subfamily == "F05_TwoLayer_ConjunctiveSiphon_Subtree")
    led5 = DecisionProvenanceLedger(run_id="t5", task_id=ds_w2_f05.task_id, arm="Hegel_M5_LiveGPT", model="m", provider_backend="mock")
    p5 = led5.record_decision(decision_type="D0", parent_decision_id=None, evidence_inputs={}, candidates_evaluated=[], chosen_option={}, rejection_reasons={}, rationale_summary="init")
    _, brief_w2_f05, _ = _build_arm_system_and_initial_brief(
        ds_w2_f05, "Hegel_M5_LiveGPT", init_rows=ds_w2_f05.pilot_rows, forest=forest,
        priors_bundle=priors_bundle, node_archive=node_archive, hippo_idx=hippo, flat_idx=flat,
        ledger=led5, parent_id=p5,
    )
    assert "GRAY-ZONE GATE BLOCKED" not in brief_w2_f05["shared_base_subtree_guidance"]["branch_rule"]
    assert len(brief_w2_f05["hegel_ppr_authorized_source_notes"]) >= 2


def test_v18_4_quota_pause_exception_and_dry_runner(tmp_path: Path) -> None:
    class _QuotaHitMockClient:
        model = "gpt-6.2-sol"
        provider_backend = "mock_quota_test"

        def call_tool_step(self, _messages):
            raise SubscriptionQuotaPausedError("usage limit reached; resets in 2h")

    forest, priors_bundle, node_archive = build_live_k45_forest_and_archive()
    ds = build_hegel_discovery_240_waves(20261121)["wave_1"][0]
    ws = tmp_path / "ep_quota"
    with pytest.raises(SubscriptionQuotaPausedError):
        run_single_live_gpt_m5_episode(
            ds,
            "Hegel_M5_LiveGPT",
            client=_QuotaHitMockClient(),
            workspace_dir=ws,
            forest=forest,
            priors_bundle=priors_bundle,
            hippo_idx=HippoRAGWormholeIndex(forest),
            flat_idx=FlatEqualMetadataIndex(forest),
            node_archive=node_archive,
        )
    assert not (ws / "decision_provenance_dag.json").exists()

    dry = run_v18_4_hegel_discovery_240_study(tmp_path / "dry_out", run_live_gpt=False)
    assert dry["total_tasks"] == 240
    assert dry["initial_forest_K"] == 45
