"""M2 Open Python Workspace & AST Wormhole Integration Tests (Hegel V16.8)."""

from __future__ import annotations

from pathlib import Path

from hegel_machine.cognition.open_workspace_labs import make_m2_open_workspace_suite
from hegel_machine.cognition.open_workspace_wormhole import (
    OpenWorkspaceFlatEqualContent,
    OpenWorkspaceHExASkills,
    OpenWorkspaceHegelResearcher,
    OpenWorkspaceStrongOrdinary,
    canonicalize_python_model_ast,
    validate_workspace_model_on_rows,
)


def test_m2_ast_canonicalizer_unifies_lechatelier_and_lenz_python_models() -> None:
    """Different variable names, opposite signs (+1 vs -1), and offsets yield identical AST skeleton!"""
    lechatelier_py = (
        "import math\n"
        "_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'\n"
        "_CORE_LAW_EXPR = '(0.95 * f[\"pressure_step\"] + 1.35 * f[\"pressure_step\"] * math.exp(-f[\"settle_time\"] / 1.8))'\n"
        "def predict(row):\n"
        "    f = row['features']\n"
        "    return 0.0 + (1.0) * (1.0) * (0.95 * f['pressure_step'] + 1.35 * f['pressure_step'] * math.exp(-f['settle_time'] / 1.8))\n"
    )
    lenz_py = (
        "import math\n"
        "_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'\n"
        "_CORE_LAW_EXPR = '(0.82 * f[\"flux_step\"] + 1.18 * f[\"flux_step\"] * math.exp(-f[\"window_dt\"] / 1.45))'\n"
        "def predict(row):\n"
        "    f = row['features']\n"
        "    return 3.0 + (-1.0) * (1.5) * (0.82 * f['flux_step'] + 1.18 * f['flux_step'] * math.exp(-f['window_dt'] / 1.45))\n"
    )
    c_chem = canonicalize_python_model_ast(lechatelier_py)
    c_lenz = canonicalize_python_model_ast(lenz_py)
    assert c_chem["ast_skeleton_digest"] == c_lenz["ast_skeleton_digest"]
    assert c_chem["feature_role_order"] == ["pressure_step", "settle_time"]
    assert c_lenz["feature_role_order"] == ["flux_step", "window_dt"]


def test_m2_subprocess_execution_error_isolated_from_scientific_refutation(tmp_path: Path) -> None:
    """Runtime crash or SyntaxError in workspace `.py` file returns EXECUTION_ERROR cleanly."""
    suite = make_m2_open_workspace_suite(9001)
    _, (fac, _) = suite["sources"][0]
    lab = fac()
    ws = tmp_path / "crash_ws"
    ws.mkdir(parents=True, exist_ok=True)
    rows = lab.acquire(
        [{"pressure_step": 2.0, "settle_time": 1.0, "prior_extent": 0.0}],
        workspace=ws,
    )
    assert rows is not None

    # Write broken Python script (ZeroDivisionError)
    (ws / "broken.py").write_text("def predict(row):\n    return 1.0 / 0.0\n", encoding="utf-8")
    crash_res = validate_workspace_model_on_rows(ws, "broken.py", rows, lab)
    assert crash_res["execution_error"] is True
    assert crash_res["outcome_status"] == "EXECUTION_ERROR"


def test_m2_end_to_end_open_workspace_formation_revision_and_10_targets(tmp_path: Path) -> None:
    """Full M2 open-workspace run: real `.py` files, subprocess evaluation, 10/10 strict pass, & 2-hop guard!"""
    suite = make_m2_open_workspace_suite(9002)

    hegel = OpenWorkspaceHegelResearcher(allow_horizontal_wormholes=True)
    src_h = hegel.form_and_revise_in_workspaces("m2_test_9002", suite, tmp_path / "hegel_src")
    assert len(src_h) == 10  # 7 empirical sources + 2 near-misses + 1 scale calibration probe
    assert len(hegel.forest.horizontal_wormhole_edges) >= 5  # 3 AST wormholes + 2 revision wormholes + empirical hop2!
    assert len(hegel.revision_events) == 2

    flat_arm = OpenWorkspaceFlatEqualContent(hegel, transitive_equivalence=True)

    ordinary_all = OpenWorkspaceStrongOrdinary(allow_archive_reconstruction=True, retain_all_validated_priors=True)
    src_o = ordinary_all.run_source_and_nearmiss_phase(suite, tmp_path / "ord_src")
    assert len(src_o) == 9
    assert len(ordinary_all.shared_python_files) == 9

    ordinary_compressed = OpenWorkspaceStrongOrdinary(allow_archive_reconstruction=True, retain_all_validated_priors=False)
    ordinary_compressed.raw_archive = list(ordinary_all.raw_archive)
    ordinary_compressed.shared_python_files = dict(ordinary_all.shared_python_files)

    hexa = OpenWorkspaceHExASkills()
    hexa.run_source_and_nearmiss_phase(suite, tmp_path / "hexa_src")

    h_meas_total = 0.0
    o_comp_meas_total = 0.0
    for _fam, t_name, (fac, ev_fn) in suite["targets"]:
        ws_h = tmp_path / "targets" / f"h_{t_name}"
        lab_h = fac()
        sub_h = hegel.solve_target(t_name, lab_h, ws_h)
        assert (ws_h / "model.py").is_file(), f"Missing authored model.py for {t_name}"
        ev_h = ev_fn(ws_h, sub_h, lab_h.observations)
        assert ev_h["strict_pass"] is True, f"Hegel M2 failed on {t_name}: {ev_h}"
        assert ev_h["false_confirmation"] is False
        h_meas_total += lab_h.meter.meas_spent

        ws_oa = tmp_path / "targets" / f"oa_{t_name}"
        lab_oa = fac()
        sub_oa = ordinary_all.solve_target(t_name, lab_oa, ws_oa)
        assert ev_fn(ws_oa, sub_oa, lab_oa.observations)["strict_pass"] is True

        ws_oc = tmp_path / "targets" / f"oc_{t_name}"
        lab_oc = fac()
        sub_oc = ordinary_compressed.solve_target(t_name, lab_oc, ws_oc)
        ev_oc = ev_fn(ws_oc, sub_oc, lab_oc.observations)
        assert ev_oc["strict_pass"] is True, f"Ordinary compressed failed on {t_name}: {ev_oc}"
        o_comp_meas_total += lab_oc.meter.meas_spent

    assert h_meas_total < o_comp_meas_total

    # Verify 2-hop non-transitive chain error check saves wasted trial vs transitive Flat_EqualContent on T_M2C4!
    t_2hop_pair = next(pair for _f, tn, pair in suite["targets"] if tn == "T_M2C4_two_hop_drift_non_transitive")
    fac_2h, ev_2h = t_2hop_pair
    lab_h2, lab_f2 = fac_2h(), fac_2h()
    ws_h2, ws_f2 = tmp_path / "twohop_h", tmp_path / "twohop_f"
    sub_h2 = hegel.solve_target("T_M2C4", lab_h2, ws_h2)
    sub_f2 = flat_arm.solve_target("T_M2C4", lab_f2, ws_f2)
    assert ev_2h(ws_h2, sub_h2, lab_h2.observations)["strict_pass"] is True
    assert ev_2h(ws_f2, sub_f2, lab_f2.observations)["strict_pass"] is True
    assert any("rejected_uncertified_two_hop" in str(c.get("decision", "")) for c in hegel.alignment_checks)
    assert lab_h2.meter.snapshot()["composite_cost"] < lab_f2.meter.snapshot()["composite_cost"]


def test_m2_counterfactual_dedup_single_source_and_program_mutation(tmp_path: Path) -> None:
    """Verifies M2 counterfactual integrity: duplicate source block, 2-source vs 1-source, & program_steps edit."""
    from copy import deepcopy

    suite = make_m2_open_workspace_suite(9005)

    # 1. Duplicate source domain must NOT fabricate an AST HorizontalWormholeEdge!
    dup_suite = deepcopy(suite)
    dup_suite["sources"] = [suite["sources"][0], suite["sources"][0]]
    dup_suite["near_misses"] = []
    dup_h = OpenWorkspaceHegelResearcher()
    dup_h.form_and_revise_in_workspaces("dup_m2", dup_suite, tmp_path / "dup_ws")
    assert len(dup_h.forest.horizontal_wormhole_edges) == 0

    # 2. 2-Source Contrastive AST Induction beats 1-Source on unseen cross-domain Queue Backpressure target!
    h_2src = OpenWorkspaceHegelResearcher()
    h_2src.form_and_revise_in_workspaces("m2_2src", suite, tmp_path / "ws_2src", single_source_only=False)

    h_1src = OpenWorkspaceHegelResearcher()
    h_1src.form_and_revise_in_workspaces("m2_1src", suite, tmp_path / "ws_1src", single_source_only=True)

    _, _, (fac_a1, ev_a1) = suite["targets"][0]  # T_M2A1_queue_backpressure_reset
    lab_2, lab_1 = fac_a1(), fac_a1()
    ws_2, ws_1 = tmp_path / "t_a1_2src", tmp_path / "t_a1_1src"
    sub_2 = h_2src.solve_target("T_M2A1", lab_2, ws_2)
    sub_1 = h_1src.solve_target("T_M2A1", lab_1, ws_1)
    assert ev_a1(ws_2, sub_2, lab_2.observations)["strict_pass"] is True
    assert ev_a1(ws_1, sub_1, lab_1.observations)["strict_pass"] is True
    assert lab_2.meter.meas_spent < lab_1.meter.meas_spent

    # 3. Mutating declarative `program_steps` on the recalled node measurably changes target probe count!
    h_mut = deepcopy(h_2src)
    for node in h_mut.forest.nodes.values():
        node["program_steps"] = [{"op": "calibrate_observation_lens"}]  # Remove scoped Python synthesis op!
    lab_mut = fac_a1()
    ws_mut = tmp_path / "t_a1_mut"
    sub_mut = h_mut.solve_target("T_M2A1", lab_mut, ws_mut)
    assert ev_a1(ws_mut, sub_mut, lab_mut.observations)["strict_pass"] is True
    assert lab_2.meter.meas_spent < lab_mut.meter.meas_spent
