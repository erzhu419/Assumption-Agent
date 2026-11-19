"""V17.0 M3 HippoRAG Specificity-Weighted PPR & Large-Scale ($K=36$) Wormhole Forest Tests.

Verifies:
1. Zero-leakage observable `sig:*` query validation (blocking `schema:*`, `domain:*`, etc.).
2. Probability conservation (`sum(p) == 1.0`) and controlled associative separation of
   `HippoRAGWormholeIndex` vs equal-metadata `Flat_BM25`, `Flat_Cosine`, and `VerticalPPR_NoWormholes`.
3. Recall != Authorization (`Hegel_M3_HippoPPR` vs `HippoPPR_UncheckedRecall`) and honest
   `scoped_unknown` exit (`false_confirmation == False`) on out-of-scope tasks (`OOS01`, `OOS02`).
4. End-to-end $K=36$ episode formation (22 Python mechanisms) and coordinate-permutation invariance.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pytest

from hegel_machine.cognition.hipporag_wormhole_index import (
    FlatEqualMetadataIndex,
    HippoRAGWormholeIndex,
    validate_observable_query_signatures,
)
from hegel_machine.cognition.m3_scale_universe import (
    M3ForestResearcher,
    make_m3_large_scale_suite,
)
from hegel_machine.cognition.wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    TaskView,
    TwoStageWormholeGate,
)


def test_v17_0_zero_leakage_observable_query_signature_guard() -> None:
    valid = validate_observable_query_signatures(["sig:arity_2", "sig:calibrated_inverted_port", "sig:arity_2"])
    assert valid == ["sig:arity_2", "sig:calibrated_inverted_port"]

    for forbidden in (
        "schema:liebig_capped_min_2d",
        "domain:unseen_datacenter",
        "mech:haldane",
        "truth:ringing",
        "oracle:hint",
        "raw_unprefixed_token",
    ):
        with pytest.raises(ValueError, match="Forbidden non-observable token"):
            validate_observable_query_signatures(["sig:arity_2", forbidden])


def test_v17_0_controlled_associative_wormhole_ppr_beats_equal_metadata_flat_and_vertical_only() -> None:
    """Exact single-factor graph test on identical node metadata (`Tree_B(-1) <-> Tree_A(+1) -> Child_Rev(+1)`)."""
    forest = PhilosophicalForest()

    def reg(nid: str, tid: str, schema: str, sign: int, sigs: list[str]) -> None:
        forest.register_node(
            node_id=nid,
            tree_id=tid,
            layer_role="mid",
            task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", 0.05),
            observation_map=ObservationMap(
                raw_channel=f"{nid}_ch",
                unit_scale=1.0,
                sign_convention=sign,
                parameter_semantics={"active_arity": 2, "schema_tag": schema, "ast_skeleton_digest": f"dig_{schema}"},
            ),
            claim_contract=ClaimContract(
                claim_id=f"c_{nid}",
                family_plugin="m3_open_workspace_ast",
                preconditions=[],
                conclusion_or_action={"schema_tag": schema},
                protocol_spec={},
                tolerance=0.05,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=sigs,
        )

    # Tree A (+1): coarse parent + witness-refined child
    reg("mid_A_parent", "tree_A", "coarse_law", +1, ["sig:arity_2", "sig:2d_symmetric_axes", "sig:2d_ridge_bottleneck"])
    reg("mid_A_child_rev", "tree_A", "refined_capped_law", +1, ["sig:arity_2", "sig:2d_symmetric_axes", "sig:2d_ridge_bottleneck", "sig:high_corner_sublinear"])
    forest.add_local_implication_edge(
        stronger_node_id="mid_A_child_rev",
        weaker_node_id="mid_A_parent",
        preserved_contract={"reduces_to": "coarse_law"},
        scope={},
        evidence_status="SUPPORTED_IN_SCOPE",
    )

    # Tree B (-1): inverted/shifted wormhole partner of coarse_law
    reg(
        "mid_B_partner_inv",
        "tree_B",
        "coarse_law",
        -1,
        ["sig:arity_2", "sig:calibrated_inverted_port", "sig:calibrated_shifted_offset", "sig:2d_symmetric_axes", "sig:2d_ridge_bottleneck"],
    )
    forest.add_horizontal_wormhole(
        wormhole_id="wh_AB",
        node_a_id="mid_A_parent",
        node_b_id="mid_B_partner_inv",
        probe_signature_refs=["sig:2d_ridge_bottleneck"],
        role_and_observation_map={"schema_tag": "coarse_law"},
        hop_error_budget=0.015,
        permission_tier="local_solver_guidance",
        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
    )

    # Single-domain inverted distractors matching 5/6 cues directly (including inverted + shifted + high_corner_sublinear!)
    for i in range(1, 4):
        reg(
            f"mid_D{i}_distractor_inv",
            f"tree_D{i}",
            f"distractor_law_{i}",
            -1,
            ["sig:arity_2", "sig:calibrated_inverted_port", "sig:calibrated_shifted_offset", "sig:2d_symmetric_axes", "sig:high_corner_sublinear"],
        )
    # Background trees (including 2 3D-style bottleneck domains sharing `sig:2d_ridge_bottleneck`, df=5)
    for j in range(1, 3):
        reg(
            f"mid_BG_kink_{j}",
            f"tree_BG_kink_{j}",
            f"bg_kink_law_{j}",
            +1,
            ["sig:arity_3", "sig:2d_ridge_bottleneck"],
        )
    for j in range(1, 7):
        reg(
            f"mid_BG{j}",
            f"tree_BG{j}",
            f"bg_law_{j}",
            +1,
            ["sig:arity_2", "sig:bg_unrelated_regime"],
        )

    query = [
        "sig:arity_2",
        "sig:calibrated_inverted_port",
        "sig:calibrated_shifted_offset",
        "sig:2d_symmetric_axes",
        "sig:2d_ridge_bottleneck",
        "sig:high_corner_sublinear",
    ]
    tv = {"goal_kind": "open_workspace_mechanism_prediction", "active_arity": 2}

    hippo = HippoRAGWormholeIndex(forest, allow_horizontal_wormholes=True)
    vert_only = HippoRAGWormholeIndex(forest, allow_horizontal_wormholes=False)
    bm25 = FlatEqualMetadataIndex(forest, mode="bm25")
    cosine = FlatEqualMetadataIndex(forest, mode="idf_cosine")

    r_hippo = hippo.retrieve(query, task_view=tv, top_k=5)
    r_vert = vert_only.retrieve(query, task_view=tv, top_k=5)
    r_bm25 = bm25.retrieve(query, task_view=tv, top_k=5)
    r_cos = cosine.retrieve(query, task_view=tv, top_k=5)

    assert abs(r_hippo["stationary_mass_sum"] - 1.0) < 1e-6
    assert abs(r_vert["stationary_mass_sum"] - 1.0) < 1e-6

    # After parent `coarse_law` is excluded by witness boundary, check rank of `refined_capped_law` vs distractors!
    non_parent_hippo = [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in r_hippo["candidates"] if c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] != "coarse_law"]
    non_parent_vert = [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in r_vert["candidates"] if c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] != "coarse_law"]
    non_parent_bm25 = [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in r_bm25["candidates"] if c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] != "coarse_law"]
    non_parent_cos = [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in r_cos["candidates"] if c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] != "coarse_law"]

    assert non_parent_hippo[0] == "refined_capped_law"
    assert non_parent_bm25[0].startswith("distractor_law_")
    assert non_parent_cos[0].startswith("distractor_law_")
    assert non_parent_vert[0].startswith("distractor_law_")


def test_v17_0_m3_36_episode_formation_recall_vs_authorization_and_oos_honesty(tmp_path: Path) -> None:
    """End-to-end verification of K=36 formation, contract gating vs UncheckedRecall, permutation invariance, and OOS."""
    suite = make_m3_large_scale_suite(6_027_101)
    gate = TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=16)
    hegel = M3ForestResearcher(gate=gate, retrieval_mode="hipporag_ppr")
    src_logs = hegel.form_and_revise_36_episodes("pytest_m3", suite, tmp_path / "src")

    ep_rows = [r for r in src_logs if r["phase"] in ("source", "source_nearmiss")]
    assert len(ep_rows) == 36
    assert all(r["strict_pass"] for r in ep_rows)
    distinct_schemas = {r["selected_schema"] for r in ep_rows}
    assert len(distinct_schemas) >= 18
    assert len(hegel.revision_events) == 4
    assert hegel.index is not None
    assert hegel.index.edge_counts["vertical_tree_edges"] == 4
    assert hegel.index.edge_counts["horizontal_wormhole_edges"] >= 10

    arms = {
        m: hegel.clone_with_mode(m)
        for m in ("hipporag_ppr", "flat_bm25", "flat_cosine", "vertical_ppr_no_wh", "hipporag_unchecked")
    }

    # 1. Out-of-scope honesty controls (`OOS01`, `OOS02`) must return `scoped_unknown` with `false_confirmation == False`
    for oname, (ofac, oev) in suite["out_of_scope_controls"]:
        lab_o = ofac()
        ws_o = tmp_path / "oos" / oname
        sub_o = arms["hipporag_ppr"].solve_target(oname, lab_o, ws_o)
        ev_o = oev(ws_o, sub_o, lab_o.observations)
        assert sub_o["status"] == "scoped_unknown"
        assert sub_o["model_path"] is None
        assert ev_o["false_confirmation"] is False

    # 2. Coordinate-permutation invariance on unseen associative target `T01`
    _, t01_name, (t01_fac, t01_ev) = suite["targets"][0]
    lab_orig = t01_fac()
    sub_orig = deepcopy(arms["hipporag_ppr"]).solve_target(t01_name, lab_orig, tmp_path / "t01_orig")
    ev_orig = t01_ev(tmp_path / "t01_orig", sub_orig, lab_orig.observations)

    lab_rev = t01_fac()
    lab_rev.feature_names = tuple(reversed(lab_rev.feature_names))
    lab_rev.task_view["feature_names"] = list(lab_rev.feature_names)
    sub_rev = deepcopy(arms["hipporag_ppr"]).solve_target(t01_name, lab_rev, tmp_path / "t01_rev")
    ev_rev = t01_ev(tmp_path / "t01_rev", sub_rev, lab_rev.observations)

    assert ev_orig["strict_pass"] is True and ev_rev["strict_pass"] is True
    assert lab_orig.meter.meas_spent == lab_rev.meter.meas_spent
    assert lab_orig.meter.compute_spent == lab_rev.meter.compute_spent

    # 3. Recall != Authorization on T05 (precondition guard), T13 (2-hop rollover bound), and T14 (ScaleMap block)
    target_map = {tname: (fac, ev_fn) for _cat, tname, (fac, ev_fn) in suite["targets"]}
    checked_cost = 0.0
    unchecked_cost = 0.0
    checked_wrong = 0
    unchecked_wrong = 0
    for key_t in (
        "T05_assoc_reordered_tcp_backpressure",
        "T13_guard_two_hop_strict_rollover",
        "T14_guard_micro_crest_scale_block",
    ):
        fac, ev_fn = target_map[key_t]
        for mode_k in ("hipporag_ppr", "hipporag_unchecked"):
            ag = arms[mode_k]
            lb = fac()
            ws = tmp_path / mode_k / key_t
            sb = ag.solve_target(key_t, lb, ws)
            ex = ev_fn(ws, sb, lb.observations)
            assert ex["strict_pass"] is True
            assert ex.get("false_confirmation", False) is False
            rlog = ag.retrieval_logs[-1]
            if mode_k == "hipporag_ppr":
                checked_cost += lb.meter.snapshot()["composite_cost"]
                checked_wrong += rlog.get("wrong_candidate_trials", 0)
            else:
                unchecked_cost += lb.meter.snapshot()["composite_cost"]
                unchecked_wrong += rlog.get("wrong_candidate_trials", 0)

    assert checked_wrong < unchecked_wrong
    assert checked_cost < unchecked_cost
