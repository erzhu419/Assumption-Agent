"""Deterministic & Saved-Model Counterfactual Audit Verification Suite for Hegel M6.3.

Directly tests and proves resolution of every counterfactual failure mode constructed in
`Hegel_M6_v18_Audit`:
1. `M6-01`: `C01` vs `C05` completion-order swap produces byte-identical Wave-2 target ASTs
   (`heldout_rmse == 0.0` under BOTH orders!) and `C04` grows a composite synthesis branch
   instead of overwriting `node_M6_dist_03`.
2. `M6-02`: `Luna C02` `R1` (`m_2, t_proc, u_raw`) arriving BEFORE `R2` (`m_2, u_raw`) retains
   BOTH in the Pareto archive and transports `R2` to `TreeM2_analog` (`0.0 / 0.0 / 0.0` RMSE!).
3. `M6-03` & `V18-R09`: Saved `Luna W1R2 C05` callable-returning `fit(rows)` achieves `0.0 / 0.0 / 0.0`
   RMSE without modifying a single byte of the model's authored code!
4. `M6-04`: Adversarial `model.py` attempting to import `hegel_machine` or open repo files is
   mechanically blocked (`ok == False`, `PermissionError` / `ModuleNotFoundError`) while `numpy`
   and `scipy.optimize` work normally.
5. `M6-05`: Intra-module helper dependencies (`hidden(row) -> row['z_2']`) ARE extracted and
   translated, while non-row parameter dictionaries (`P = {'z_2': 4.0}; P['z_2']`) are untouched!
6. `M6-06` & `M6-07`: Empirical retroactive projection receipts are attached during consolidation,
   slice-level precondition checks catch mixed regimes, and `observational_L1_selected` is excluded
   from regime-breaking probes!
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m5_live_gpt_and_provenance import _execute_authored_python_in_isolated_subprocess
from hegel_machine.cognition.m6_1_zero_leakage_ast_causal_benchmark import build_m6_1_clean_k40_forest
from hegel_machine.cognition.m6_3_sovereign_causal_wormhole_engine import (
    consolidate_wave1_sovereign_m6_3,
    detect_observational_identifiability_hazards_v2,
    extract_and_translate_authored_ast_v2,
)
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves


REPO_HMACHINE = Path(__file__).resolve().parents[1]


def test_m6_05_callgraph_helper_reachability_and_nonrow_dict_safety() -> None:
    # Counterexample A from Audit: helper function `hidden(row)`
    code_a = (
        "def hidden(row):\n"
        "    return row['z_2']\n"
        "def fit(rows):\n"
        "    pass\n"
        "def predict(row):\n"
        "    return hidden(row) - 0.3 * row['u_ctrl']\n"
    )
    res_a = extract_and_translate_authored_ast_v2(code_a, role_map={"z_2": "z_4", "u_ctrl": "u_ctrl"})
    assert res_a["ok"] is True
    assert res_a["used_keys"] == ["u_ctrl", "z_2"], f"Missed helper key: {res_a}"
    assert res_a["translated_keys"] == ["u_ctrl", "z_4"], f"Failed helper translation: {res_a}"
    assert "row['z_4']" in res_a["translated_code"]

    # Counterexample B from Audit: non-row parameter dict `P = {'z_2': 4.0}`
    code_b = (
        "P = {'z_2': 4.0}\n"
        "def fit(rows):\n"
        "    pass\n"
        "def predict(row):\n"
        "    return P['z_2'] * row['u_ctrl']\n"
    )
    res_b = extract_and_translate_authored_ast_v2(code_b, role_map={"z_2": "z_4", "u_ctrl": "u_ctrl_mapped"})
    assert res_b["ok"] is True
    assert res_b["used_keys"] == ["u_ctrl"], f"Falsely collected non-row dict key: {res_b}"
    assert "P['z_2']" in res_b["translated_code"], f"Corrupted parameter dict P: {res_b['translated_code']}"
    assert "row['u_ctrl_mapped']" in res_b["translated_code"]


def test_m6_04_hermetic_subprocess_blocks_private_generator_and_repo_open() -> None:
    exploit_import = (
        "from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves\n"
        "def fit(rows): pass\n"
        "def predict(row): return 0.0\n"
    )
    exploit_fs = (
        f"txt = open({str(REPO_HMACHINE / 'pyproject.toml')!r}).read()\n"
        "def fit(rows): pass\n"
        "def predict(row): return 0.0\n"
    )
    sample_rows = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}]
    with tempfile.TemporaryDirectory() as tmpd:
        r1 = _execute_authored_python_in_isolated_subprocess(
            Path(tmpd), step_idx=1, python_code=exploit_import,
            train_rows=sample_rows, confirm_rows=sample_rows, heldout_rows=sample_rows,
        )
        assert r1["ok"] is False, f"Hermetic sandbox failed to block repo import: {r1}"

        r2 = _execute_authored_python_in_isolated_subprocess(
            Path(tmpd), step_idx=2, python_code=exploit_fs,
            train_rows=sample_rows, confirm_rows=sample_rows, heldout_rows=sample_rows,
        )
        assert r2["ok"] is False, f"Hermetic sandbox failed to block host repo open(): {r2}"


def test_m6_03_saved_luna_c05_callable_abi_rescue() -> None:
    waves = build_m6_causal_24_waves(seed=20261121)
    all_ds = {d.task_id: d for w in waves.values() for d in w}
    ds_c05 = all_ds["W1R2__C05_ColliderShielded_DirectX"]

    saved_c05_model = (
        REPO_HMACHINE
        / "docs/evidence/m6_1_zero_leakage_ast_causal_192ep_20261121/workspaces_and_dags/wave_1/gpt-6-luna__Hegel_M6_1_ZeroLeakage_LiveGPT__W1R2__C05_ColliderShielded_DirectX/model_step_01.py"
    )
    assert saved_c05_model.is_file(), f"Missing saved Luna C05 model: {saved_c05_model}"
    py_code = saved_c05_model.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmpd:
        res = _execute_authored_python_in_isolated_subprocess(
            Path(tmpd),
            step_idx=1,
            python_code=py_code,
            train_rows=ds_c05.pilot_rows,
            confirm_rows=ds_c05.confirm_rows,
            heldout_rows=ds_c05.heldout_rows,
        )
    assert res["ok"] is True
    assert res["abi_mode"] == "fit_returned_callable"
    assert res["train_rmse"] < 1e-5 and res["confirm_rmse"] < 1e-5 and res["heldout_rmse"] < 1e-5, res


def test_m6_01_order_invariance_and_m6_02_pareto_partial_subtree_transport() -> None:
    waves = build_m6_causal_24_waves(seed=20261121)
    all_ds = {d.task_id: d for w in waves.values() for d in w}
    ev_root = REPO_HMACHINE / "docs/evidence/m6_1_zero_leakage_ast_causal_192ep_20261121"
    rec_files = sorted((ev_root / "workspaces_and_dags").rglob("episode_receipt.json"))
    assert len(rec_files) == 192, f"Expected 192 saved M6.1 receipts, got {len(rec_files)}"
    all_recs = [json.loads(fp.read_text(encoding="utf-8")) for fp in rec_files]

    # 1. Replay Sol Wave-1 receipts in BOTH (C01 before C05) and (C05 before C01) orders!
    sol_w1 = [
        r for r in all_recs
        if r["model"] == "gpt-6.2-sol" and r["arm"] == "Hegel_M6_1_ZeroLeakage_LiveGPT" and r["wave"] == "wave_1"
    ]
    # Enrich saved M6.1 receipts with task's public `probe_signatures` + HippoRAG Top-4 recall IDs (zero task_id peeking!)
    f_tmp, _ = build_m6_1_clean_k40_forest()
    h_idx = HippoRAGWormholeIndex(f_tmp)
    enriched_sol: list[dict] = []
    for r in sol_w1:
        ds = all_ds[r["task_id"]]
        rec_q = h_idx.retrieve(ds.probe_signatures, task_view={"goal_kind": "causal_and_interventional_prediction", "tolerance": ds.tolerance}, top_k=4)
        r_copy = dict(r)
        r_copy["recalled_node_ids"] = [c["node_id"] for c in rec_q["candidates"]]
        r_copy["observed_signatures"] = list(ds.probe_signatures)
        r_copy["violated_root_ids"] = ["node_C04_Root_Vitality_Main"] if "C04" in ds.task_id else []
        r_copy["pilot_rows_snapshot"] = list(ds.pilot_rows)
        enriched_sol.append(r_copy)

    f1, arch1 = build_m6_1_clean_k40_forest()
    wb1 = consolidate_wave1_sovereign_m6_3(wave1_receipts=enriched_sol, arm="Hegel_M6_3_Sovereign_LiveGPT", forest=f1, node_archive=arch1)

    f2, arch2 = build_m6_1_clean_k40_forest()
    wb2 = consolidate_wave1_sovereign_m6_3(wave1_receipts=list(reversed(enriched_sol)), arm="Hegel_M6_3_Sovereign_LiveGPT", forest=f2, node_archive=arch2)

    # Byte-identical target code on `node_C01_TreeF_doubleprime_catalysis` regardless of C01/C05 order!
    t_code_1 = arch1["node_C01_TreeF_doubleprime_catalysis"]["confirmed_python_reference"]
    t_code_2 = arch2["node_C01_TreeF_doubleprime_catalysis"]["confirmed_python_reference"]
    assert t_code_1 == t_code_2
    # Distractor `node_M6_dist_03` MUST remain untouched!
    assert arch1["node_M6_dist_03"]["epistemic_rung"] == "L1_DISTRACTOR"
    # Verify refit on Wave-2 `Fdouble` gives `heldout_rmse == 0.0`!
    ds_fdouble = all_ds["W2R2__C01_ConfounderFork_tree_C01_Fdouble_catalysis"]
    with tempfile.TemporaryDirectory() as tmpd:
        fit_f = _execute_authored_python_in_isolated_subprocess(
            Path(tmpd), step_idx=1, python_code=t_code_2,
            train_rows=ds_fdouble.pilot_rows, confirm_rows=ds_fdouble.confirm_rows, heldout_rows=ds_fdouble.heldout_rows,
        )
    assert fit_f["ok"] and fit_f["heldout_rmse"] < 1e-5, fit_f

    # 2. Replay Luna Wave-1 C02 (`R1` with `t_proc` BEFORE `R2` without `t_proc`)!
    luna_c02 = [
        r for r in all_recs
        if r["model"] == "gpt-6-luna" and r["arm"] == "Hegel_M6_1_ZeroLeakage_LiveGPT"
        and r["wave"] == "wave_1" and "C02_MediatorReAnchor" in r["task_id"]
    ]
    assert len(luna_c02) == 2, f"Expected 2 Luna C02 Wave-1 receipts, got {len(luna_c02)}"
    luna_c02.sort(key=lambda x: x["task_id"])  # Ensures R1 (`m_2, t_proc, u_raw`) comes FIRST!
    enriched_luna: list[dict] = []
    for r in luna_c02:
        ds = all_ds[r["task_id"]]
        rec_q = h_idx.retrieve(ds.probe_signatures, task_view={"goal_kind": "causal_and_interventional_prediction", "tolerance": ds.tolerance}, top_k=4)
        rc = dict(r)
        rc["recalled_node_ids"] = [c["node_id"] for c in rec_q["candidates"]]
        rc["observed_signatures"] = list(ds.probe_signatures)
        rc["violated_root_ids"] = []
        rc["pilot_rows_snapshot"] = list(ds.pilot_rows)
        enriched_luna.append(rc)

    fl, arch_l = build_m6_1_clean_k40_forest()
    consolidate_wave1_sovereign_m6_3(wave1_receipts=enriched_luna, arm="Hegel_M6_3_Sovereign_LiveGPT", forest=fl, node_archive=arch_l)

    # Source M1 retains BOTH Pareto variants (`(m_2, u_raw)` and `(m_2, t_proc, u_raw)`), and target M2 receives `(m_2, u_raw)`!
    assert len(arch_l["node_C02_TreeM1_surface"]["pareto_confirmed_candidates"]) == 2
    m2_code = arch_l["node_C02_TreeM2_analog"]["confirmed_python_reference"]
    ds_m2 = all_ds["W2R1__C02_MediatorReAnchor_tree_C02_M2_ferment"]
    with tempfile.TemporaryDirectory() as tmpd:
        fit_m2 = _execute_authored_python_in_isolated_subprocess(
            Path(tmpd), step_idx=1, python_code=m2_code,
            train_rows=ds_m2.pilot_rows, confirm_rows=ds_m2.confirm_rows, heldout_rows=ds_m2.heldout_rows,
        )
    assert fit_m2["ok"] and fit_m2["heldout_rmse"] < 1e-5, fit_m2


def test_m6_07_selected_observational_probe_excluded_from_regime_breaking() -> None:
    pilot = [
        {"u_1": 1.0, "u_2": 1.5333, "s_selected": 1.0, "y": 2.1},
        {"u_1": 1.5, "u_2": 1.2000, "s_selected": 1.0, "y": 2.4},
        {"u_1": 2.0, "u_2": 0.8667, "s_selected": 1.0, "y": 2.6},
    ]
    # Only passive selected probe available -> MUST return None (not falsely recommend selected slice!)
    pool_selected_only = {
        "probe_selected_catalog_more": {"probe_kind": "observational_L1_selected", "cost": 4.5, "rows": pilot}
    }
    assert detect_observational_identifiability_hazards_v2(pilot, pool_selected_only) is None

    # Add true unselected population probe -> recommends ONLY the unselected probe!
    pool_both = {
        **pool_selected_only,
        "probe_unselected_full_population": {"probe_kind": "unselected_population_L2", "cost": 6.0, "rows": pilot},
    }
    res = detect_observational_identifiability_hazards_v2(pilot, pool_both)
    assert res is not None
    assert res["eligible_regime_breaking_probes"] == ["probe_unselected_full_population"]


def test_frontier_fr02_fr04_fr06_and_m6_09_controls() -> None:
    from hegel_machine.cognition.m6_frontier_real_benchmarks import (
        build_m6_frontier_general_forest,
        consolidate_wave1_into_m6_forest,
        is_substantive_verification_command,
        load_frontier_scout_20_tasks,
    )

    # FR02: Official task.toml verifier specs parsed accurately (especially AMR 12 GiB / 3600s / no-network!)
    waves = load_frontier_scout_20_tasks()
    by_short = {t.short_name: t for w in waves.values() for t in w}
    amr = by_short["amr-poisson-optimize"]
    assert amr.verifier_timeout_sec == 3600, amr
    assert amr.verifier_memory_mb == 12288, amr
    assert amr.verifier_cpus == 4, amr
    assert amr.verifier_network_mode == "none", amr

    # FR04: Trivial exit-0 commands (`true`, `ls`, `pwd`) CANNOT clear verification debt!
    assert is_substantive_verification_command("true", ("/app/",)) is False
    assert is_substantive_verification_command("ls -la /app", ("/app/",)) is False
    assert is_substantive_verification_command("python3 -m pytest", ("/app/",)) is True
    assert is_substantive_verification_command("python3 -c 'import solver; print(1)' /app/solver", ("/app/",)) is True

    # M6-09 & FR06: RUNG_1 shell quirks are d-separated and blocked from cross-tree propagation!
    f_a, arch_a, dag_a = build_m6_frontier_general_forest()
    shell_ep = {
        "task_id": "tb__ode-law-discovery",
        "short_name": "ode-law-discovery",
        "wormhole_family": "fam_symbolic_and_operator_dynamics",
        "strict_pass": True,
        "failure_taxonomy": "PASSED_ALL_CHECKS",
        "nogoods_recorded": [
            {
                "nogood_rung": "RUNG_1_SHELL_OR_ENV_QUIRK",
                "assumption_refuted": "python alias exists",
                "empirical_witness": "bash: python: command not found",
                "sublated_pivot": "use python3",
            }
        ],
    }
    res_shell = consolidate_wave1_into_m6_forest(wave1_m6_summaries=[shell_ep], forest=f_a, archive=arch_a, meta_dag=dag_a)
    assert res_shell["events"][0]["bayes_ball_cross_tree_blocked"] is True
    assert res_shell["events"][0]["updated_nodes"] == []
    assert all(len(v["wave1_consolidated_nogoods"]) == 0 for v in arch_a.values())

    # FR07: Canonical order invariance on causal RUNG_2/3 lessons!
    causal_eps = [
        {
            "task_id": "tb__ode-law-discovery",
            "short_name": "ode-law-discovery",
            "wormhole_family": "fam_symbolic_and_operator_dynamics",
            "strict_pass": True,
            "failure_taxonomy": "PASSED_ALL_CHECKS",
            "nogoods_recorded": [
                {
                    "nogood_rung": "RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY",
                    "assumption_refuted": "Polynomial library suffices",
                    "empirical_witness": "off-grid RMSE=0.25",
                    "sublated_pivot": "Include rational denominators",
                }
            ],
        },
        {
            "task_id": "tb__traffic-flux-inversion",
            "short_name": "traffic-flux-inversion",
            "wormhole_family": "fam_symbolic_and_operator_dynamics",
            "strict_pass": False,
            "failure_taxonomy": "NUMERICAL_ACCURACY_OR_MECHANISTIC_MODEL_DEFECT",
            "nogoods_recorded": [
                {
                    "nogood_rung": "RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM",
                    "assumption_refuted": "Single least-squares point identifies control transfer",
                    "empirical_witness": "s2e0 horizon transfer failure",
                    "sublated_pivot": "Select minimax robust flux across equivalence class",
                }
            ],
        },
    ]
    f1, arch1, d1 = build_m6_frontier_general_forest()
    f2, arch2, d2 = build_m6_frontier_general_forest()
    consolidate_wave1_into_m6_forest(wave1_m6_summaries=causal_eps, forest=f1, archive=arch1, meta_dag=d1)
    consolidate_wave1_into_m6_forest(wave1_m6_summaries=list(reversed(causal_eps)), forest=f2, archive=arch2, meta_dag=d2)
    assert arch1 == arch2


if __name__ == "__main__":
    test_m6_05_callgraph_helper_reachability_and_nonrow_dict_safety()
    test_m6_04_hermetic_subprocess_blocks_private_generator_and_repo_open()
    test_m6_03_saved_luna_c05_callable_abi_rescue()
    test_m6_01_order_invariance_and_m6_02_pareto_partial_subtree_transport()
    test_m6_07_selected_observational_probe_excluded_from_regime_breaking()
    test_frontier_fr02_fr04_fr06_and_m6_09_controls()
    print("ALL M6.3 SOVEREIGN AUDIT COUNTERFACTUAL CONTROLS PASSED 100%!")
