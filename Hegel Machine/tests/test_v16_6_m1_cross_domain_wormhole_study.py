"""M1 End-to-End & Causal Intervention Tests for Hegel V16.6 Cross-Domain Wormholes."""

from __future__ import annotations

import json
from pathlib import Path

from hegel_machine.cognition.cross_domain_arms import (
    HExAInspiredSkillResearcher,
    HegelV166Researcher,
    StrongOrdinaryResearcher,
)
from hegel_machine.cognition.cross_domain_clusters import (
    make_cluster_a_suite,
    make_cluster_b_suite,
    make_cluster_c_same_service_suite,
)


def test_m1_autonomous_formation_nearmiss_revision_and_horizontal_wormholes() -> None:
    """Verifies autonomous 2-source formation, ALIGN/CEGAR revisions, and strict TaskView safety."""
    ca = make_cluster_a_suite(7001)
    cb = make_cluster_b_suite(7001)
    cc = make_cluster_c_same_service_suite(7001)

    hegel = HegelV166Researcher(allow_horizontal_wormholes=True)
    hegel.form_and_revise_from_experience("test_cluster_7001", ca, cb)

    # 1. Verify local trees & horizontal mid-layer wormholes formed autonomously
    assert "wh_A_ci_dac_bisect" in hegel.forest.horizontal_wormhole_edges
    assert "wh_B_sys_bio_rational" in hegel.forest.horizontal_wormhole_edges
    assert "wh_B_sys_to_multi_bottleneck" in hegel.forest.horizontal_wormhole_edges
    assert "ci_leaf_monotone_earliest" in hegel.forest.nodes
    assert hegel.forest.nodes["ci_mid_bisect"]["version"] == 2
    # Weak root `sys_root_concave_limit` was honestly demoted after sqrt near-miss!
    assert hegel.forest.nodes["sys_root_concave_limit"]["utility_ledger"]["archived_low_utility"] is True

    # 2. Verify all 5 Cluster A targets pass strictly with ZERO illegal goal relaxation
    for t_name, (fac, ev_fn) in ca["targets"]:
        task = fac()
        sub = hegel.solve_target(t_name, task)
        ev = ev_fn(sub)
        assert ev["strict_pass"] is True, f"Cluster A failed on {t_name}: {ev}"
        assert ev.get("false_confirmation") is False

    # 3. Verify all 6 Cluster B targets + 3 Cluster C same-service multi-view targets pass strictly
    hegel_meas_by_target: dict[str, float] = {}
    for t_name, (fac, ev_fn) in [*cb["targets"], *cc["targets"]]:
        task = fac()
        sub = hegel.solve_target(t_name, task)
        ev = ev_fn(sub)
        assert ev["strict_pass"] is True, f"Cluster B/C failed on {t_name}: {ev}"
        assert ev.get("false_confirmation") is False
        hegel_meas_by_target[t_name] = task.meter.meas_spent

    # 4. Verify net probe savings on T_B1 (rational wormhole) & T_B3 (CEGAR tied joint probe) vs O_strong
    ord_arm = StrongOrdinaryResearcher()
    ord_arm.run_source_and_nearmiss_phase(ca, cb)
    hexa_arm = HExAInspiredSkillResearcher()
    hexa_arm.run_source_and_nearmiss_phase(ca, cb)

    tb1_fac, tb1_eval = dict(cb["targets"])["T_B1_unseen_rational_transport"]
    to_b1 = tb1_fac()
    assert tb1_eval(ord_arm.solve_target("T_B1", to_b1))["strict_pass"]
    assert hegel_meas_by_target["T_B1_unseen_rational_transport"] < to_b1.meter.meas_spent

    tb3_fac, tb3_eval = dict(cb["targets"])["T_B3_tied_bottleneck_joint"]
    th_b3, to_b3, tx_b3 = tb3_fac(), tb3_fac(), tb3_fac()
    assert tb3_eval(hegel.solve_target("T_B3", th_b3))["strict_pass"]
    assert tb3_eval(ord_arm.solve_target("T_B3", to_b3))["strict_pass"]
    assert tb3_eval(hexa_arm.solve_target("T_B3", tx_b3))["strict_pass"]
    assert th_b3.meter.meas_spent < to_b3.meter.meas_spent
    assert th_b3.meter.meas_spent < tx_b3.meter.meas_spent

    # 5. Verify ScaleMap blocks coarse window_average on burst p99 tail latency T_C2 (saving probes vs O_HExA!)
    tc2_fac, tc2_eval = dict(cc["targets"])["T_C2_service_burst_tail_latency"]
    th_c2, tx_c2 = tc2_fac(), tc2_fac()
    assert tc2_eval(hegel.solve_target("T_C2", th_c2))["strict_pass"]
    assert tc2_eval(hexa_arm.solve_target("T_C2", tx_c2))["strict_pass"]
    assert th_c2.meter.meas_spent < tx_c2.meter.meas_spent
