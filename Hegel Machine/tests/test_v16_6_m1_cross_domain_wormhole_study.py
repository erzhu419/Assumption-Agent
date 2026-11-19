"""M1 End-to-End & Counterfactual Red-Team Tests for Hegel V16.7 Cross-Domain Wormholes."""

from __future__ import annotations

from copy import deepcopy

from hegel_machine.cognition.cross_domain_arms import (
    FlatEqualContentResearcher,
    HExAInspiredSkillResearcher,
    HegelV166Researcher,
    StrongOrdinaryResearcher,
)
from hegel_machine.cognition.cross_domain_clusters import (
    CostMeter,
    PublicOrderedBoundaryTask,
    make_cluster_a_suite,
    make_cluster_b_suite,
    make_cluster_c_same_service_suite,
)
from hegel_machine.cognition.wormhole_forest import compose_wormhole_chain


def test_m1_autonomous_formation_nearmiss_revision_and_horizontal_wormholes() -> None:
    """Verifies autonomous 2-source formation, ALIGN/CEGAR revisions, empowered baselines, & ScaleMap safety."""
    ca = make_cluster_a_suite(7001)
    cb = make_cluster_b_suite(7001)
    cc = make_cluster_c_same_service_suite(7001)

    hegel = HegelV166Researcher(allow_horizontal_wormholes=True)
    src_logs = hegel.form_and_revise_from_experience("test_cluster_7001", ca, cb, cc)
    flat_arm = FlatEqualContentResearcher(hegel)

    # 1. Verify full raw archive retention, live endpoint version coherence, & real QuestionSpecs
    assert len(hegel.forest.raw_episode_archive) == len(src_logs) == 8
    assert len(hegel.forest.questions) >= 5
    assert "wh_A_ci_dac_bisect" in hegel.forest.horizontal_wormhole_edges
    assert "wh_B_sys_bio_rational" in hegel.forest.horizontal_wormhole_edges
    assert "wh_B_sys_to_multi_bottleneck" in hegel.forest.horizontal_wormhole_edges
    assert "ci_leaf_monotone_earliest" in hegel.forest.nodes
    assert hegel.forest.nodes["ci_mid_bisect"]["version"] == 2
    assert hegel.forest.horizontal_wormhole_edges["wh_A_ci_dac_bisect"]["node_a_version"] == 2
    assert hegel.forest.nodes["sys_root_concave_limit"]["utility_ledger"]["trials"] == 3
    assert hegel.forest.nodes["sys_root_concave_limit"]["utility_ledger"]["archived_low_utility"] is True

    # 2. Verify all 5 Cluster A + 6 Cluster B + 3 Cluster C targets pass strictly with ZERO false confirmations
    hegel_meas_by_target: dict[str, float] = {}
    hegel_comp_by_target: dict[str, float] = {}
    for t_name, (fac, ev_fn) in [*ca["targets"], *cb["targets"], *cc["targets"]]:
        task = fac()
        sub = hegel.solve_target(t_name, task)
        ev = ev_fn(sub)
        assert ev["strict_pass"] is True, f"Failed on {t_name}: {ev}"
        assert ev.get("false_confirmation") is False
        hegel_meas_by_target[t_name] = task.meter.meas_spent
        hegel_comp_by_target[t_name] = task.meter.snapshot()["composite_cost"]

    # 3. Verify O_strong genuinely benefits from raw_archive vs empty archive, while Hegel still wins overall!
    ord_empty = StrongOrdinaryResearcher()
    ord_arm = StrongOrdinaryResearcher()
    ord_arm.run_source_and_nearmiss_phase(ca, cb)
    hexa_arm = HExAInspiredSkillResearcher()
    hexa_arm.run_source_and_nearmiss_phase(ca, cb)

    tb1_fac, tb1_eval = dict(cb["targets"])["T_B1_unseen_rational_transport"]
    to_b1_empty, to_b1_hist = tb1_fac(), tb1_fac()
    assert tb1_eval(ord_empty.solve_target("T_B1", to_b1_empty))["strict_pass"]
    assert tb1_eval(ord_arm.solve_target("T_B1", to_b1_hist))["strict_pass"]
    # Empowered O_strong caches rational_saturation from raw history -> improves from 42.0 to 24.0 meas!
    assert ord_arm.rebuilt_cache["1d_prior_family"] == "rational_saturation"
    assert to_b1_hist.meter.meas_spent < to_b1_empty.meter.meas_spent
    # Hegel avoids the 28-unit target-time reconstruction compute tax on T_B1:
    assert hegel_comp_by_target["T_B1_unseen_rational_transport"] < to_b1_hist.meter.snapshot()["composite_cost"]

    # On 2D tied bottleneck T_B3, Hegel saves physical measurement probes against BOTH empowered O_strong and O_HExA!
    tb3_fac, tb3_eval = dict(cb["targets"])["T_B3_tied_bottleneck_joint"]
    th_b3, to_b3, tx_b3 = tb3_fac(), tb3_fac(), tb3_fac()
    assert tb3_eval(hegel.solve_target("T_B3", th_b3))["strict_pass"]
    assert tb3_eval(ord_arm.solve_target("T_B3", to_b3))["strict_pass"]
    assert tb3_eval(hexa_arm.solve_target("T_B3", tx_b3))["strict_pass"]
    assert th_b3.meter.meas_spent < to_b3.meter.meas_spent
    assert th_b3.meter.meas_spent < tx_b3.meter.meas_spent

    # 4. Flat_equal_content matches 1-hop probes while compose_wormhole_chain blocks 2-hop non-transitive drift!
    tf_b1 = tb1_fac()
    assert tb1_eval(flat_arm.solve_target("T_B1", tf_b1))["strict_pass"]
    assert tf_b1.meter.meas_spent == hegel_meas_by_target["T_B1_unseen_rational_transport"]

    hop1 = {"wormhole_id": "wh_1", "hop_error_budget": 0.018, "lipschitz_factor": 1.0, "permission_tier": "local_solver_guidance"}
    hop2 = {"wormhole_id": "wh_2", "hop_error_budget": 0.018, "lipschitz_factor": 1.0, "permission_tier": "local_solver_guidance"}
    chain_2hop = compose_wormhole_chain([hop1, hop2], target_tolerance=0.025)
    assert chain_2hop["within_tolerance"] is False
    assert chain_2hop["effective_permission_tier"] == "bounded_trial"


def test_m1_v167_short_early_pocket_boundary_soundness() -> None:
    """Reproduces & verifies fix for `probe_boundary_contract.py` short early anomaly pockets."""
    ca = make_cluster_a_suite(3026101)
    cb = make_cluster_b_suite(3026101)
    h = HegelV166Researcher()
    h.form_and_revise_from_experience("audit_boundary", ca, cb)
    o = StrongOrdinaryResearcher()

    cases = {
        "opposite_endpoints_short_early_pocket": [1 if i in (3, 4) or i >= 80 else 0 for i in range(128)],
        "same_endpoints_short_early_pocket": [1 if i in (3, 4) or 40 <= i < 56 else 0 for i in range(128)],
    }
    for case_name, seq in cases.items():
        for arm_runner in (h, o):
            task = PublicOrderedBoundaryTask(
                task_id=case_name,
                surface_domain="audit_release_history",
                length=len(seq),
                cost_per_query=3.0,
                task_view={
                    "goal_kind": "earliest_global_threshold",
                    "target_metric": "exact_adjacent_boundary",
                    "tolerance": 0.0,
                    "input_domain": {"type": "ordered_discrete", "length": len(seq)},
                    "stochastic_oracle_hint": False,
                },
                meter=CostMeter(obs_budget=72, meas_budget=216.0, compute_budget=500.0),
                private_oracle_fn=lambda i, values=seq: values[i],
            )
            res = arm_runner.solve_target(case_name, task)
            assert res["status"] == "completed"
            assert res["boundary_right_index"] == 3

    # Also verify budget-starved sequence NEVER fabricates an unverified earliest index!
    seq_late = [0 if i < 90 else 1 for i in range(128)]
    starved = PublicOrderedBoundaryTask(
        task_id="starved_earliest",
        surface_domain="audit_release_history",
        length=128,
        cost_per_query=3.0,
        task_view={"goal_kind": "earliest_global_threshold", "target_metric": "exact_adjacent_boundary", "tolerance": 0.0},
        meter=CostMeter(obs_budget=20, meas_budget=60.0, compute_budget=500.0),
        private_oracle_fn=lambda i: seq_late[i],
    )
    res_starved = h.solve_target("starved_earliest", starved)
    assert res_starved["status"] != "completed"
    assert res_starved["boundary_right_index"] is None


def test_m1_v167_counterfactual_formation_and_behavioral_interventions() -> None:
    """Reproduces & verifies all counterfactual controls from `probe_formation_and_ablations.py`."""
    seed = 3026101
    ca = make_cluster_a_suite(seed)
    cb = make_cluster_b_suite(seed)
    targets = dict(cb["targets"])

    # 1. Constant-zero failed A sources must NOT produce SUPPORTED_IN_SCOPE claims or wormholes!
    ca_failed = deepcopy(ca)
    ca_failed["sources"] = [
        (
            name,
            (
                (lambda orig=fac: (lambda t=orig(): (setattr(t, "_oracle", lambda _i: 0) or t))()),
                ev_fn,
            ),
        )
        for name, (fac, ev_fn) in ca["sources"]
    ]
    failed_h = HegelV166Researcher()
    failed_h.form_and_revise_from_experience("failed_a", ca_failed, cb)
    assert "wh_A_ci_dac_bisect" not in failed_h.forest.horizontal_wormhole_edges
    assert failed_h.forest.node_history["ci_mid_bisect"][0]["claim_contract"]["evidence_status"] == "NOT_DETECTED_UNDER_PROTOCOL"
    assert failed_h.forest.nodes["dac_mid_bisect"]["claim_contract"]["evidence_status"] == "NOT_DETECTED_UNDER_PROTOCOL"

    # 2. Duplicate identical B source must NOT fabricate a second domain tree or wormhole!
    cb_dup = deepcopy(cb)
    cb_dup["sources"] = [cb["sources"][0], cb["sources"][0]]
    dup_h = HegelV166Researcher()
    dup_h.form_and_revise_from_experience("dup_b", ca, cb_dup)
    assert "wh_B_sys_bio_rational" not in dup_h.forest.horizontal_wormhole_edges
    assert "bio_mid_rational" not in dup_h.forest.nodes

    # 3. Capped 2D near-miss `min(2*u1, 2*u2, 10)` vs target `16` must mark homogeneous scaling REFUTED_IN_SCOPE!
    cb_capped = deepcopy(cb)
    nm_name, (nm_fac, _) = cb["near_misses"][0]

    def capped_fac():
        t = nm_fac()
        t._sim = lambda ctrl: min(2.0 * ctrl["u1"], 2.0 * ctrl["u2"], 10.0)
        return t

    def capped_eval(sub):
        rec = sub.get("recommended_controls", {})
        ach = min(2.0 * rec.get("u1", 0.0), 2.0 * rec.get("u2", 0.0), 10.0)
        ok = sub.get("status") == "completed" and ach >= 16.0 * 0.97
        return {"strict_pass": ok, "false_confirmation": sub.get("status") == "completed" and not ok}

    cb_capped["near_misses"][0] = (nm_name, (capped_fac, capped_eval))
    cap_h = HegelV166Researcher()
    cap_h.form_and_revise_from_experience("cap_b1", ca, cb_capped)
    assert cap_h.forest.nodes["bot_mid_limiting_factor_cegar"]["claim_contract"]["evidence_status"] == "REFUTED_IN_SCOPE"
    assert "wh_B_sys_to_multi_bottleneck" not in cap_h.forest.horizontal_wormhole_edges
    assert cap_h.forest.nodes["bot_root_joint_complementarity_question"]["claim_contract"]["evidence_status"] == "SUPPORTED_IN_SCOPE"

    # 4. Mutating declarative `program_steps` AND removing learned ALIGN exclusion must change target traces!
    base_h = HegelV166Researcher()
    base_h.form_and_revise_from_experience("base_ctl", ca, cb)

    # (a) Remove ALIGN exclusion on T_B5_unbounded_sqrt_concave -> wastes bounded trial confirmation probe!
    no_excl_h = deepcopy(base_h)
    no_excl_h.forest.horizontal_wormhole_edges["wh_B_sys_bio_rational"]["align_conditions"]["exclusion_predicates"] = []
    tb5_fac, _ = targets["T_B5_unbounded_sqrt_concave"]
    t_with, t_without = tb5_fac(), tb5_fac()
    base_h.solve_target("T_B5", t_with)
    no_excl_h.solve_target("T_B5", t_without)
    assert t_with.meter.meas_spent < t_without.meter.meas_spent

    # (b) Mutate declarative `program_steps` on `bot_mid_limiting_factor_cegar` -> changes probe behavior on T_B3!
    mut_prog_h = deepcopy(base_h)
    mut_prog_h.forest.nodes["bot_mid_limiting_factor_cegar"]["program_steps"] = [
        {"op": "probe_single_axes", "step_delta": 1.0}  # Joint step op removed!
    ]
    tb3_fac, _ = targets["T_B3_tied_bottleneck_joint"]
    t_prog_full, t_prog_mut = tb3_fac(), tb3_fac()
    base_h.solve_target("T_B3", t_prog_full)
    mut_prog_h.solve_target("T_B3", t_prog_mut)
    assert t_prog_full.meter.meas_spent < t_prog_mut.meter.meas_spent

    # 5. Deleting O_HExA_inspired's learned skill/mistake entries changes its target trajectory!
    hexa_full = HExAInspiredSkillResearcher()
    hexa_full.run_source_and_nearmiss_phase(ca, cb)
    hexa_stripped = deepcopy(hexa_full)
    hexa_stripped.skill_bank.clear()
    hexa_stripped.mistake_bank.clear()
    tb1_fac, _ = targets["T_B1_unseen_rational_transport"]
    tx_full, tx_stripped = tb1_fac(), tb1_fac()
    hexa_full.solve_target("T_B1", tx_full)
    hexa_stripped.solve_target("T_B1", tx_stripped)
    assert tx_full.meter.meas_spent < tx_stripped.meter.meas_spent
