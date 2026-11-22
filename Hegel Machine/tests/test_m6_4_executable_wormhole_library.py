"""Unit & Counterfactual Verification Tests for Hegel M6.4 Executable Wormhole Library (`Skill∞` + Pure AST Sublation)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

HEGEL_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = HEGEL_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    NODE_TO_EXECUTABLE_WORMHOLE_MODULE,
    SKILL_INIT_PY,
    consolidate_wave1_executable_modules_into_forest,
    extract_pure_reusable_functions_from_python,
    harvest_container_skill_activations,
    stage_recalled_executable_wormholes_in_container,
    sum_skill_activation_calls,
)
from hegel_machine.cognition.m6_frontier_real_benchmarks import (
    FrontierBenchmarkTask,
    build_m6_frontier_general_forest,
    exec_container_bash,
    extract_public_task_signatures,
    load_frontier_scout_20_tasks,
    run_biomystery_verifier,
    start_detached_container,
    stop_container_safely,
)


def test_all_six_executable_wormhole_modules_and_skill_telemetry() -> None:
    with tempfile.TemporaryDirectory(prefix="test_m64_pkg_") as tmp_s:
        tmp_p = Path(tmp_s)
        act_log = tmp_p / ".test_activations.jsonl"
        pkg_dir = tmp_p / "hegel_wormhole_skills"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        init_src = SKILL_INIT_PY.replace('"/tmp/.hegel_skill_activations.jsonl"', json.dumps(str(act_log)))
        (pkg_dir / "__init__.py").write_text(init_src, encoding="utf-8")

        written_mods: set[str] = set()
        for fname, src, _ in NODE_TO_EXECUTABLE_WORMHOLE_MODULE.values():
            if fname not in written_mods:
                # Verify zero private leakage tokens in foundational modules
                for bad in ("/tests", "97703", "0.236", "answer_rubric", "11, 42, 55"):
                    assert bad not in src, f"Forbidden token {bad} found in {fname}"
                (pkg_dir / fname).write_text(src, encoding="utf-8")
                written_mods.add(fname)
        assert len(written_mods) == 6

        sys.path.insert(0, str(tmp_p))
        try:
            m1 = importlib.import_module("hegel_wormhole_skills.symbolic_and_dynamical_wormhole")
            m2 = importlib.import_module("hegel_wormhole_skills.conservation_and_hybrid_relay_wormhole")
            m3 = importlib.import_module("hegel_wormhole_skills.variational_assimilation_and_certificate_wormhole")
            m4 = importlib.import_module("hegel_wormhole_skills.cohort_transfer_and_imputation_wormhole")
            m5 = importlib.import_module("hegel_wormhole_skills.wave_dispersion_and_lopo_imaging_wormhole")
            m6 = importlib.import_module("hegel_wormhole_skills.stage_protocol_and_outlier_calibration_wormhole")

            rng = np.random.default_rng(123)
            # 1. Symbolic & dynamical scanner + OOF Macro-F1
            X_sym = rng.uniform(-1.5, 1.5, size=(240, 12))
            logits = 2.6 * np.cos(1.5 * (X_sym[:, 2] - X_sym[:, 7])) + 2.0 * np.tanh(X_sym[:, 1] * X_sym[:, 4] * X_sym[:, 9])
            y_sym = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(int)
            scan = m1.null_calibrated_symbolic_scanner(X_sym, y_sym, max_triples_cols=12, n_null_trials=60)
            assert scan["pairwise"]["top_hits"][0]["exceeds_q95"] is True
            Phi_good = np.column_stack([np.cos(1.5 * (X_sym[:, 2] - X_sym[:, 7])), np.tanh(X_sym[:, 1] * X_sym[:, 4] * X_sym[:, 9])])
            oof_res = m1.oof_macro_f1_evaluator(Phi_good, y_sym)
            assert oof_res["oof_macro_f1"] > 0.88

            # 2. Conservation Godunov + parity + minimax consensus
            rho0 = np.linspace(0.2, 0.4, 20)
            inflow = np.full(15, 0.18)
            rcap = np.full(15, 0.3)
            out_rho = m2.godunov_concave_flux_rollout(rho0, inflow, rcap, 0.01, 0.05, lambda r: r * (1.0 - r), 0.5)
            assert out_rho.shape == (16, 20) and np.all((out_rho >= 0.0) & (out_rho <= 1.0))
            par = m2.verify_vectorized_vs_scalar_recurrence(lambda x: np.cumsum(x), lambda x: np.array([sum(x[: i + 1]) for i in range(len(x))]), [(np.arange(6.0),)])
            assert par["matched"] is True
            cands = [{"a": 0.1}, {"a": 0.5}, {"a": 9.0}]
            sel = m2.minimax_consensus_candidate_selector(cands, lambda cs: np.array([[0.0, 0.1], [0.05, 0.12], [4.0, 5.0]]))
            assert sel["selected_index"] in (0, 1)

            # 3. Variational Fourier basis + zero-sum box + certificate compaction
            B, rough = m3.real_fourier_smoothness_basis(16, 4)
            assert B.shape == (16, 9) and np.allclose(B.T @ B, np.eye(9), atol=1e-10)
            assert len(rough) == 9
            tau_proj = m3.project_zero_sum_box(np.array([0.08, -0.01, -0.02, 0.04]), max_abs=0.05)
            assert abs(float(np.sum(tau_proj))) < 1e-10 and float(np.max(np.abs(tau_proj))) <= 0.05 + 1e-12
            comp = m3.audit_certificate_sibling_compaction(
                {"p1": {"parent_lb": 9.995, "leaf_count": 4}, "p2": {"parent_lb": 9.50, "leaf_count": 3}},
                global_upper_bound=10.0,
                rel_gap_tol=1e-3,
            )
            assert comp["collapsible_parent_ids"] == ["p1"] and comp["potential_leaf_reduction"] == 3
            G_sim = rng.normal(size=(40, 5))
            Y_obs = G_sim * np.array([1.5, -0.8, 2.0, 0.5, -1.2]) + np.array([3.0, -1.0, 0.0, 2.5, -4.0])
            vp = m3.affine_variable_projection_residuals(G_sim, Y_obs, noise_scale=0.1)
            assert vp["worst_sensor_rms"] < 1e-10 and np.allclose(vp["gains"], [1.5, -0.8, 2.0, 0.5, -1.2])
            leaves4 = [(3, 0), (3, 1), (3, 2), (3, 3)]
            it_comp = m3.iterated_binary_cube_sibling_compactor(leaves4, lambda cands: [True] * len(cands))
            assert it_comp["final_leaves"] == 1 and it_comp["passes_run"] == 2

            # 4. Cross-cohort shift aligner + LOCO balanced-accuracy CV
            cohorts = np.repeat(["C1", "C2", "C3"], 60)
            z_lat = rng.normal(0.0, 1.0, size=(180, 4))
            y_clin = (z_lat[:, 0] - 0.8 * z_lat[:, 1] > 0.2).astype(int)
            X_shifted = z_lat.copy()
            X_shifted[cohorts == "C2"] += 4.5  # strong batch offset!
            X_shifted[cohorts == "C3"] = X_shifted[cohorts == "C3"] * 2.5 - 3.0
            X_aligned, align_info = m4.cohort_shift_aligner(X_shifted, cohorts, method="robust_mad")
            assert align_info["cohort_median_shift_rms"]["C2"] > 1.0
            from sklearn.linear_model import LogisticRegression

            loco = m4.leave_one_cohort_out_calibrated_cv(
                X_aligned,
                y_clin,
                cohorts,
                lambda Xtr, ytr, Xte, ctr, cte: LogisticRegression().fit(Xtr, ytr).predict_proba(Xte)[:, 1],
            )
            assert loco["worst_cohort_balanced_accuracy"] > 0.85

            # 5. Wave Hilbert envelope + LOPO grid stability
            t = np.linspace(0.0, 1.0, 256, endpoint=False)
            carrier = np.exp(-((t - 0.4) ** 2) / 0.005) * np.cos(2.0 * np.pi * 24.0 * t)
            env = m5.analytic_hilbert_envelope(carrier)
            assert abs(float(np.max(env)) - 1.0) < 0.05
            gx = np.stack(np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11)), axis=-1).reshape(-1, 2)
            d2_grid = np.sum((gx - np.array([[0.4, 0.6]])) ** 2, axis=1)
            pair_maps = np.tile(np.exp(-20.0 * d2_grid), (8, 1))
            lopo = m5.leave_one_pair_out_grid_stability(pair_maps, gx)
            assert lopo["lopo_max_shift"] == 0.0 and np.allclose(lopo["peak_xy"], [0.4, 0.6])

            # 6. Hold-to-unload breakpoint + Mahalanobis innovation scores
            sig = np.concatenate([np.ones(25), 1.0 - 0.04 * np.arange(1, 20)])
            bp = m6.detect_hold_to_unload_breakpoint(sig, nominal_end_idx=22, half_window=5, ramp_len=6)
            assert bp["unload_start_idx"] == 24
            s_el = m6.feng_ngan_creep_corrected_stiffness(100.0, 0.5, -500.0)
            assert abs(float(s_el) - (1.0 / (0.01 + 0.001))) < 1e-9
            nis = m6.mahalanobis_innovation_scores(np.array([[2.0, 0.0], [0.0, 3.0]]), np.tile(np.eye(2), (2, 1, 1)))
            assert np.allclose(nis, [4.0, 9.0])

            # Verify @track_wormhole_skill runtime activation JSONL telemetry!
            records = [json.loads(line) for line in act_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            called_fns = {r["function"] for r in records}
            assert len(records) >= 14
            assert {
                "oof_macro_f1_evaluator",
                "null_calibrated_symbolic_scanner",
                "godunov_concave_flux_rollout",
                "real_fourier_smoothness_basis",
                "cohort_shift_aligner",
                "leave_one_pair_out_grid_stability",
                "detect_hold_to_unload_breakpoint",
            }.issubset(called_fns)
        finally:
            sys.path.remove(str(tmp_p))
            for k in [k for k in sys.modules if k == "hegel_wormhole_skills" or k.startswith("hegel_wormhole_skills.")]:
                del sys.modules[k]


def test_pure_ast_extraction_and_wave1_to_wave2_sublation() -> None:
    mixed_wave1_src = '''
import numpy as np

SECRET_GLOBAL = 42.0

def impure_uses_global(x):
    """Should be rejected because SECRET_GLOBAL is an unbound module global!"""
    return np.asarray(x, dtype=float) * SECRET_GLOBAL + 1.0

def impure_reads_app_path(x):
    """Should be rejected because it references /app/data!"""
    with open("/app/data/train.csv") as fp:
        return fp.read()

def pure_huber_weights(residuals, delta=1.345):
    """Pure self-contained IRLS Huber weight helper safe for cross-task sublation."""
    r = np.abs(np.asarray(residuals, dtype=float))
    return np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
'''
    extracted = extract_pure_reusable_functions_from_python(mixed_wave1_src)
    assert [name for name, _ in extracted] == ["pure_huber_weights"]

    forest, archive, _ = build_m6_frontier_general_forest()
    with tempfile.TemporaryDirectory(prefix="test_w1_sub_") as tmp_s:
        ep_dir = Path(tmp_s) / "ep_nav"
        snap_dir = ep_dir / "04_submitted_artifacts_snapshot"
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "calibrate.py").write_text(mixed_wave1_src, encoding="utf-8")

        events = consolidate_wave1_executable_modules_into_forest(
            wave1_episode_dirs=[
                (
                    {
                        "task_id": "tb__navigation-sensor-calibration",
                        "short_name": "navigation-sensor-calibration",
                        "strict_pass": True,
                        "wormhole_family": "fam_robust_control_and_calibration",
                    },
                    ep_dir,
                )
            ],
            forest=forest,
            archive=archive,
        )
        assert len(events) == 1
        assert events[0]["exported_functions"] == ["pure_huber_weights"]
        assert "node_F08_robust_calibration_and_safety_control" in events[0]["propagated_to_nodes"]
        # Verify cross-wormhole propagation onto signature-overlapping neighbor node!
        assert len(events[0]["propagated_to_nodes"]) >= 2


def test_biomystery_list_match_false_falls_through_and_docker_staging() -> None:
    # 1. Verify official `scorer.py` parity: `_list_match` returning False must NOT set `det_method == 'list_match'`!
    t_bm = FrontierBenchmarkTask(
        task_id="bm__hb024",
        short_name="hb024",
        benchmark="BioMysteryBench-Preview",
        wave="wave_2",
        scientific_domain="Genomics",
        wormhole_family="fam_biomystery_genomic_homology",
        instruction_text="Identify tissue groups and host species.",
        agent_image_tag="hegel_biomystery_slim:v2",
        biomystery_answer_rubric="2 tissue groups: cloacal, mouth; 3 host species: Boiga dendrophila, Laticauda laticaudata, Trimeresurus flavomaculatus",
    )
    rep = run_biomystery_verifier(
        t_bm,
        "2 tissue groups: Cloacal and Mouth. 3 host species: Boiga dendrophila, Laticauda laticaudata, and Trimeresurus flavomaculatus.",
        codex_info={"available": False},
    )
    assert rep["deterministic_method"] == "none"

    # 2. Verify real Docker container staging + Skill∞ runtime execution + JSONL harvesting!
    _, archive, _ = build_m6_frontier_general_forest()
    cid = start_detached_container("tb_sci_symbolic-regression:v01", memory_limit="1024m", cpus="2")
    try:
        manifest = stage_recalled_executable_wormholes_in_container(
            cid,
            [archive["node_F01_symbolic_sparse_dynamics"], archive["node_F07_clinical_and_network_imputation"]],
        )
        assert len(manifest) == 2
        ex = exec_container_bash(
            cid,
            "PYTHONPATH=/tmp python3 -c \"import numpy as np; "
            "from hegel_wormhole_skills.symbolic_and_dynamical_wormhole import null_calibrated_symbolic_scanner; "
            "res = null_calibrated_symbolic_scanner(np.random.randn(60, 5), (np.random.rand(60)>0.5).astype(int), n_null_trials=20); "
            "print('OK', res['univariate']['null_q95'])\"",
            timeout_sec=30,
        )
        assert ex["exit_code"] == 0 and "OK" in ex["stdout"], ex
        acts = harvest_container_skill_activations(cid)
        assert sum_skill_activation_calls(acts) == 1 and acts[0]["function"] == "null_calibrated_symbolic_scanner"
    finally:
        stop_container_safely(cid)


def test_reentrant_heavy_timing_rwlock_no_deadlock() -> None:
    scripts_dir = HEGEL_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_m6_5_frontier_28turn_study import _HeavyTimingRWLock

    lk = _HeavyTimingRWLock()
    lk.acquire_read()
    lk.release_read()
    lk.acquire_write()
    try:
        # Exact reentrant path (`run_tb_separate_verifier` -> `exec_container_bash`) that deadlocked previously!
        lk.acquire_read()
        lk.release_read()
    finally:
        lk.release_write()
    lk.acquire_read()
    lk.release_read()


def test_word_boundary_public_signatures_prevent_ode_substring_false_positives() -> None:
    waves = load_frontier_scout_20_tasks()
    by_id = {t.task_id: t for w in waves.values() for t in w}
    vb_sigs = extract_public_task_signatures(by_id["tb__virtual-baseline-localization"])
    assert "sig:guided_wave_dispersion_and_geometry_transfer" in vb_sigs
    assert "sig:symbolic_ode_and_law_regression" not in vb_sigs
    nano_sigs = extract_public_task_signatures(by_id["tb__nanoindentation-property-extraction"])
    assert "sig:robust_physical_calibration_and_safety_margins" in nano_sigs
    assert "sig:symbolic_ode_and_law_regression" not in nano_sigs


if __name__ == "__main__":
    test_all_six_executable_wormhole_modules_and_skill_telemetry()
    test_pure_ast_extraction_and_wave1_to_wave2_sublation()
    test_biomystery_list_match_false_falls_through_and_docker_staging()
    print("ALL M6.4 EXECUTABLE WORMHOLE UNIT & COUNTERFACTUAL TESTS PASSED 100%!")
