"""V17.1 (M3.1) Audit Closure, Retrieval Factorization & Genuine Multi-Hop Walk Tests.

Directly regression-tests EVERY technical witness from `hegel_M3_Audit` PLUS proves where
multi-hop graph random walk (`alpha > 0` across `HorizontalWormholeEdge` + `local_implication_edge`)
strictly outperforms `PPR(alpha=0, WormholePool)`, `Flat_BM25_WormholePool`, `Flat_BM25`,
and `VerticalPPR_NoWormholes`!
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m3_1_walk_factorization import (
    M3_1FactorizedResearcher,
    make_m3_1_factorized_walk_suite,
)
from hegel_machine.cognition.m3_scale_universe import (
    M3ForestResearcher,
    M3HExACodeSkills,
    _active_and_fixed_features,
    _estimate_observation_lens,
    build_m3_pilot_points,
    make_m3_large_scale_suite,
    run_m3_workspace_synthesizer,
)
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.open_python_revision import (
    ScriptedOpenGrammarCodeProvider,
    _run_code_agent_with_prefix_confirm,
    run_open_revision_chain_experiment,
)
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab, _evaluate_workspace_submission
from hegel_machine.cognition.open_workspace_wormhole import (
    canonicalize_python_model_ast,
    validate_workspace_model_on_rows,
)
from hegel_machine.cognition.wormhole_forest import TwoStageWormholeGate, authorize_abstraction_use


def test_v17_1_closure_witnesses_file_memorizer_ast_shadowing_feedback_and_receipts(tmp_path: Path) -> None:
    """Locks in 100% zero-defect closure of all `hegel_M3_Audit/closure/probe_closure.py` witnesses."""
    # 1. AST shadowed callable & keyword precision separation
    s1 = canonicalize_python_model_ast("def exp(x):\n    return x*x\ndef predict(row):\n    return exp(row['features']['x'])\n")
    s2 = canonicalize_python_model_ast("def exp(x):\n    return x*x*x\ndef predict(row):\n    return exp(row['features']['x'])\n")
    assert s1["skeleton_status"] == "UNSUPPORTED_AST_CONSTRUCT"
    assert s2["skeleton_status"] == "UNSUPPORTED_AST_CONSTRUCT"
    assert s1["ast_skeleton_digest"] != s2["ast_skeleton_digest"]

    k1 = canonicalize_python_model_ast("def predict(row):\n    return round(row['features']['x'], ndigits=1)\n")
    k2 = canonicalize_python_model_ast("def predict(row):\n    return round(row['features']['x'], ndigits=3)\n")
    assert k1["skeleton_status"] == "VERIFIED_PURE_AST"
    assert k2["skeleton_status"] == "VERIFIED_PURE_AST"
    assert k1["ast_skeleton_digest"] != k2["ast_skeleton_digest"]

    # 2. File-channel memorizers (`observations.json` and `model_observations.json`) blocked during confirmation!
    ws = tmp_path / "file_mem_ws"
    ws.mkdir(parents=True, exist_ok=True)
    lab = OpenWorkspaceTaskLab(
        task_id="file_iso",
        surface_domain="file_iso",
        feature_names=("u",),
        feature_bounds={"u": (1.0, 10.0)},
        cost_per_obs=5.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01},
        meter=CostMeter(obs_budget=12, meas_budget=100.0, compute_budget=300.0),
        private_oracle_fn=lambda p: p["u"] ** 2,
    )
    lab.acquire([{"u": 1.0}, {"u": 2.0}], workspace=ws, phase="presolve")
    prefix = list(lab.observations)
    for fname, target_json in [("file_mem.py", "observations.json"), ("model_file_mem.py", "model_observations.json")]:
        (ws / fname).write_text(
            f"import json\nmem={{}}\ndef fit(obs):\n    global mem\n    rows=json.load(open({target_json!r}))\n"
            "    mem={r['features']['u']: r['target'] for r in rows}\n"
            "def predict(row):\n    return mem.get(row['features']['u'], 0.0)\n",
            encoding="utf-8",
        )
    # Even if `lab.acquire(..., workspace=ws)` synced confirmation row to `ws`, sandbox isolates `prefix`!
    c_rows = lab.acquire([{"u": 3.0}], workspace=ws, phase="confirm")
    assert c_rows is not None
    for fname in ("file_mem.py", "model_file_mem.py"):
        vres = validate_workspace_model_on_rows(ws, fname, c_rows, lab, training_prefix=prefix)
        assert vres["rmse"] == 9.0

    # Full CodeAgent lifecycle with file_memorizer returns `scoped_unknown`, `false_confirmation == False`
    ws_ca = tmp_path / "ca_file_mem"
    lab_ca = OpenWorkspaceTaskLab(
        task_id="ca_iso",
        surface_domain="ca_iso",
        feature_names=("u",),
        feature_bounds={"u": (1.0, 10.0)},
        cost_per_obs=5.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01},
        meter=CostMeter(obs_budget=12, meas_budget=100.0, compute_budget=400.0),
        private_oracle_fn=lambda p: p["u"] ** 2,
    )
    prov_mem = ScriptedOpenGrammarCodeProvider(
        query_batches=[[{"u": 1.0}, {"u": 2.0}]],
        candidate_specs=[("fmem.py", (ws / "file_mem.py").read_text(encoding="utf-8"))],
        tolerance=0.01,
    )
    sub_ca = _run_code_agent_with_prefix_confirm(lab_ca, ws_ca, prov_mem, {"u": 3.0})
    ev_ca = _evaluate_workspace_submission(
        ws_ca, sub_ca.get("model_path"), lab_ca.observations, [Observation({"u": 4.0}, 16.0)], 0.01, submission_status=sub_ca["status"]
    )
    assert sub_ca["status"] == "scoped_unknown"
    assert ev_ca["strict_pass"] is False and ev_ca["false_confirmation"] is False

    # 3. Diagnostic `stdout` genuinely drives next `write` action (`same_next_action == False`)
    base_p = ScriptedOpenGrammarCodeProvider(
        query_batches=[],
        candidate_specs=[("candidate.py", "def predict(row):\n    return 1.0\n")],
        diagnostic_script="print('diag')",
    )
    _ = base_p.next_action({"events": []})
    pa, pb = deepcopy(base_p), deepcopy(base_p)
    act_a = pa.next_action({"events": [{"action": "run", "feedback": {"stdout": "high_drive_turnover_with_bypass"}}]})
    act_b = pb.next_action({"events": [{"action": "run", "feedback": {"stdout": "pure_oscillation_no_bypass"}}]})
    assert act_a != act_b

    # 4. Mandatory receipt fields & no legacy `heldout_confirmed=True` bypass on target receipts
    cc = {"claim_id": "c", "permission_tier": "local_solver_guidance", "evidence_status": "SUPPORTED_IN_SCOPE", "preconditions": []}
    stale_rec = {"claim_id": "c", "passed": True, "observed_error": 0.002, "target_id": "old_t", "node_version": 1, "model_sha256": "sha"}
    for tv_case, hflag in [
        ({"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01}, False),
        ({"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01, "target_id": "old_t", "node_version": 1, "expected_model_sha256": "sha"}, False),
        ({"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01, "target_id": "new_t"}, True),
    ]:
        assert authorize_abstraction_use(
            requested_tier="local_solver_guidance",
            task_view=tv_case,
            target_features={},
            claim_contract=cc,
            evidence_receipt=stale_rec,
            heldout_confirmed=hflag,
        )["granted"] is False

    # 5. Open revision chain has equal initial pilot measurement cost (no forced extra queries on Ordinary!)
    open_rep = run_open_revision_chain_experiment(5026101, tmp_path / "open_rev_fair")
    for _tname, arms in open_rep["downstream_targets"].items():
        assert arms["Hegel_OpenRevision_Forest"]["costs"]["measurement_cost"] == arms["Continuing_Ordinary_CodeSession"]["costs"]["measurement_cost"]
        assert arms["Hegel_OpenRevision_Forest"]["costs"]["composite_cost"] < arms["Continuing_Ordinary_CodeSession"]["costs"]["composite_cost"]


def test_v17_1_m3_36_verified_asts_monotone_pricing_program_steps_tiebreak_and_hexa_reachability(tmp_path: Path) -> None:
    """Verifies all `MECHANISM_AUDIT_M3` & `results_audit` fixes on the K=36 universe."""
    suite = make_m3_large_scale_suite(7_027_101)
    gate = TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=16)
    hegel = M3ForestResearcher(gate=gate, retrieval_mode="hipporag_ppr")
    logs = hegel.form_and_revise_36_episodes("audit_m3", suite, tmp_path / "src")

    # 1. All 36 source + near-miss episodes get `VERIFIED_PURE_AST` (0 `mismatch:` rejections!)
    ep_logs = [r for r in logs if r["phase"] in ("source", "source_nearmiss")]
    assert len(ep_logs) == 36
    assert all(r["skeleton_status"] == "VERIFIED_PURE_AST" for r in ep_logs)
    # Bridge fit cost is explicitly logged in `logs`!
    assert any(r["phase"] == "source_bridge_calibration" for r in logs)
    assert hegel.index is not None
    assert hegel.index.edge_counts["horizontal_wormhole_edges"] >= 14

    # 2. Monotone work-based synthesizer compute pricing (subset <= unrestricted, full list == unrestricted)
    _, _, (t_fac, _) = suite["targets"][5]  # 1D target `T06`
    lab_p = t_fac()
    lens_p = _estimate_observation_lens(lab_p, tmp_path / "price_pilot")
    lab_p.acquire(build_m3_pilot_points(lab_p), workspace=tmp_path / "price_pilot")
    priors_1d = sorted({r["selected_schema"] for r in ep_logs if r["active_arity"] == 1})
    lab_sub, lab_full = deepcopy(lab_p), deepcopy(lab_p)
    c0_sub, c0_full = lab_sub.meter.compute_spent, lab_full.meter.compute_spent
    run_m3_workspace_synthesizer(tmp_path / "p_sub", lab_sub, observation_lens=lens_p, allowed_schemas=priors_1d)
    run_m3_workspace_synthesizer(tmp_path / "p_full", lab_full, observation_lens=lens_p, allowed_schemas=None)
    assert (lab_sub.meter.compute_spent - c0_sub) < (lab_full.meter.compute_spent - c0_full)

    # 3. Identifier-independent tie-breaking: renaming `node_id`s produces 0 ranking differences!
    renamed_forest = deepcopy(hegel.forest)
    old_to_new = {nid: f"zz_perm_{i:03d}" for i, nid in enumerate(reversed(sorted(renamed_forest.nodes)))}
    new_nodes = {}
    for old_nid, node in renamed_forest.nodes.items():
        nn = deepcopy(node)
        nn["node_id"] = old_to_new[old_nid]
        new_nodes[nn["node_id"]] = nn
    renamed_forest.nodes = new_nodes
    for v_edge in renamed_forest.local_implication_edges:
        v_edge["stronger_node_id"] = old_to_new[v_edge["stronger_node_id"]]
        v_edge["weaker_node_id"] = old_to_new[v_edge["weaker_node_id"]]
    for wh in renamed_forest.horizontal_wormhole_edges.values():
        wh["node_a_id"] = old_to_new[wh["node_a_id"]]
        wh["node_b_id"] = old_to_new[wh["node_b_id"]]

    q_check = ["sig:arity_2", "sig:calibrated_inverted_port", "sig:calibrated_shifted_offset", "sig:2d_multi_factor_positive", "sig:2d_symmetric_axes", "sig:high_corner_sublinear"]
    tv_check = {"goal_kind": "open_workspace_mechanism_prediction", "active_arity": 2}
    for cls in (HippoRAGWormholeIndex, FlatEqualMetadataIndex):
        idx_a = cls(hegel.forest)
        idx_b = cls(renamed_forest)
        sa = [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in idx_a.retrieve(q_check, task_view=tv_check, top_k=6)["candidates"]]
        sb = [c["target_node"]["claim_contract"]["conclusion_or_action"]["schema_tag"] for c in idx_b.retrieve(q_check, task_view=tv_check, top_k=6)["candidates"]]
        assert sa == sb

    # 4. Deleting `program_steps` on recalled nodes genuinely degrades `confirmed_via_recall` and increases cost!
    _, t01_name, (t01_fac, t01_ev) = suite["targets"][0]
    h_intact = deepcopy(hegel)
    h_empty_steps = deepcopy(hegel)
    for n in h_empty_steps.forest.nodes.values():
        n["program_steps"] = []
    h_empty_steps.build_or_refresh_index()

    l_ok, l_del = t01_fac(), t01_fac()
    s_ok = h_intact.solve_target(t01_name, l_ok, tmp_path / "steps_ok")
    s_del = h_empty_steps.solve_target(t01_name, l_del, tmp_path / "steps_del")
    assert t01_ev(tmp_path / "steps_ok", s_ok, l_ok.observations)["strict_pass"] is True
    assert t01_ev(tmp_path / "steps_del", s_del, l_del.observations)["strict_pass"] is True
    assert h_intact.retrieval_logs[-1]["confirmed_via_recall"] is True
    assert h_empty_steps.retrieval_logs[-1]["confirmed_via_recall"] is False
    assert l_ok.meter.meas_spent < l_del.meter.meas_spent

    # 5. `M3HExACodeSkills` condition-matched `mistake_bank` is genuinely reachable & causal!
    hexa_full = M3HExACodeSkills()
    hexa_full.run_36_episode_source_phase(suite, tmp_path / "hexa_src")
    hexa_no_mistake = deepcopy(hexa_full)
    hexa_no_mistake.mistake_bank.clear()
    hexa_no_mistake.mistake_sigs.clear()

    lh_full, lh_nom = t01_fac(), t01_fac()
    sh_full = hexa_full.solve_target(t01_name, lh_full, tmp_path / "hx_full")
    sh_nom = hexa_no_mistake.solve_target(t01_name, lh_nom, tmp_path / "hx_nom")
    assert t01_ev(tmp_path / "hx_full", sh_full, lh_full.observations)["strict_pass"] is True
    assert t01_ev(tmp_path / "hx_nom", sh_nom, lh_nom.observations)["strict_pass"] is True
    assert lh_full.meter.meas_spent < lh_nom.meter.meas_spent


def test_v17_1_safe_pilot_multi_hop_walk_strictly_beats_alpha0_pool_and_flat_wormhole_pool(tmp_path: Path) -> None:
    """Proves genuine non-direct 2-hop graph walk (`Q_safe -> S_B <-> S_A -> NM_A`) where alpha=0 & Flat_Pool score 0!"""
    suite = make_m3_1_factorized_walk_suite(8_027_101, target_k=36)
    gate = TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=16)
    base = M3_1FactorizedResearcher(gate=gate, retrieval_mode="hipporag_ppr", damping_alpha=0.65)
    src_logs = base.form_and_revise_episodes("m3_1_walk_test", suite, tmp_path / "src_walk")
    ep_logs = [r for r in src_logs if r["phase"] in ("source", "source_nearmiss")]
    assert len(ep_logs) == 36
    assert all(r["strict_pass"] for r in ep_logs)

    modes = (
        "hipporag_ppr",            # Walk (alpha=0.65) + WormholePool
        "hipporag_alpha0_pool",    # No Walk (alpha=0.0) + WormholePool
        "flat_bm25_wormhole_pool", # Flat BM25 + WormholePool
        "flat_bm25",               # Standard Flat BM25
        "vertical_ppr_no_wh",      # Vertical Walk Only (No Horizontal Wormholes)
    )
    arms = {m: base.clone_with_mode(m) for m in modes}

    walk_recalled_counts = {m: 0 for m in modes}
    meas_totals = {m: 0.0 for m in modes}
    comp_totals = {m: 0.0 for m in modes}

    for _cat, tname, (tfac, tev) in suite["walk_targets"]:
        for m in modes:
            lb = tfac()
            ws = tmp_path / m / tname
            sub = arms[m].solve_target(tname, lb, ws)
            ev = tev(ws, sub, lb.observations)
            assert ev["strict_pass"] is True, f"{m} failed on {tname}: {ev}"
            assert ev["false_confirmation"] is False
            rlog = arms[m].retrieval_logs[-1]
            if rlog["confirmed_via_recall"] and rlog["selected_reached_only_by_walk"]:
                walk_recalled_counts[m] += 1
            meas_totals[m] += lb.meter.meas_spent
            comp_totals[m] += lb.meter.snapshot()["composite_cost"]

    # ONLY `hipporag_ppr` (alpha>0 + horizontal wormholes + vertical edges) reaches all 4 disjoint-cue walk targets!
    assert walk_recalled_counts["hipporag_ppr"] == 4
    assert walk_recalled_counts["hipporag_alpha0_pool"] == 0
    assert walk_recalled_counts["flat_bm25_wormhole_pool"] == 0
    assert walk_recalled_counts["flat_bm25"] == 0
    assert walk_recalled_counts["vertical_ppr_no_wh"] == 0

    for other in ("hipporag_alpha0_pool", "flat_bm25_wormhole_pool", "flat_bm25", "vertical_ppr_no_wh"):
        assert meas_totals["hipporag_ppr"] < meas_totals[other]
        assert comp_totals["hipporag_ppr"] < comp_totals[other]
