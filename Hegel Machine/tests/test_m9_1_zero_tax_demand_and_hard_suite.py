"""Unit and container verifier tests for `M9.1` Zero-Turn-1-Tax Demand-Triggered Retrieval & `Hegel-M91-HardDemand-2`.

Verifies:
1. `demand_triggered_zero_tax` produces `0` Turn-1 manifest bytes (`len == 2`, identical to `Bare`!), mounts all `6`
   qualified `L1` modules silently on disk, supports `--search` without importing skill modules, and produces `0`
   recommended cards on `vr04` (`Indefinite Complex Helmholtz`).
2. `M8OfflineContainerPlanner` unconfounded router parity: `Bare` and `Skill` arms with `in_container_substantive_receipt_verified == True`
   receive identical `SUPPORTED_IN_PARTIAL_SCOPE` state on non-`TB-Science` benchmarks while `m7_4` `LOTO` remains `14/20`.
3. `hd01`: Reference 3-stage solver composing `align_reference_panel` + `balanced_cell_weights` + `weighted_column_fill` +
   `cauchy_linear_fit` + `row_incidence_blocks` + `gram_diagonal_minorant` + `box_ridge_cube_lower_bound` PASSES (`strict_pass == True`),
   while naive OLS (Cauchy spike failure) FAILS (`strict_pass == False`).
4. `hd02`: Reference 3D (`21,952` unknowns) solver composing `galerkin_preconditioner` + `solve_residual_checked_cg` +
   `fit_ridge_nuisance` + `profile_candidate_amplitudes` PASSES (`strict_pass == True`, `< 14.0s`), while unpreconditioned `CG(450)`
   FAILS (`strict_pass == False`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    build_container_pre_action_state,
    load_historical_suite_episode_records,
)
from hegel_machine.cognition.m9_0_voluntary_reuse_benchmark import get_voluntary_reuse_m90a_frontier_tasks
from hegel_machine.cognition.m9_1_hard_demand_benchmark import (
    get_hard_demand_m91_frontier_tasks,
    run_hard_demand_private_verifier_by_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V5_BUNDLE_PATH = (
    REPO_ROOT
    / "docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126/pretrained_autogenous_forest_v5_m89.json"
)


def _load_v5_l1_sources() -> tuple[dict[str, dict], dict[str, str]]:
    _, v5_arch, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE_PATH)
    out: dict[str, str] = {}
    for _, rec in m7a.iter_live_archive_nodes(v5_arch):
        for m in rec.get("autogenous_wormhole_modules") or []:
            fn = str(m.get("filename") or "")
            if fn.startswith("autogenous_prim_") and fn not in out:
                out[fn] = str(m.get("python_code") or "").replace("from __future__ import annotations\n", "")
    return v5_arch, out


def _bundle_single_py_tar(td: Path, arc_rel: str, code_str: str, tar_name: str) -> Path:
    src_p = td / tar_name.replace(".tar", "_dir") / arc_rel.lstrip("/")
    src_p.parent.mkdir(parents=True, exist_ok=True)
    src_p.write_text(code_str, encoding="utf-8")
    tar_p = td / tar_name
    with tarfile.open(tar_p, "w") as tf:
        tf.add(str(src_p), arcname=arc_rel.lstrip("/"))
    return tar_p


def test_m9_1_zero_turn1_tax_manifest_search_cli_and_unconfounded_router() -> None:
    v5_arch, _ = _load_v5_l1_sources()
    vr_tasks = {t.task_id: t for t in get_voluntary_reuse_m90a_frontier_tasks()}
    hd_tasks = {t.task_id: t for t in get_hard_demand_m91_frontier_tasks()}

    # 1. `demand_triggered_zero_tax` emits `manifest == []` (`0` Turn-1 manifest bytes!) while mounting `6` L1 modules on disk!
    mods_hd01, man_hd01, diag_hd01 = m64.collect_recalled_executable_wormhole_sources(
        [],
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=v5_arch,
        task_query_text=hd_tasks["hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline"].instruction_text,
        disclosure_mode="demand_triggered_zero_tax",
        staging_policy="net_value_gated",
        return_diagnostics=True,
    )
    assert man_hd01 == []
    assert diag_hd01["recommended_turn1_l1_count"] == 0
    assert diag_hd01["mounted_on_disk_l1_count"] == 6
    assert diag_hd01["queryable_catalog_l1_count"] == 6
    assert diag_hd01["demand_triggered_recommended_l1_count"] >= 3
    assert len(diag_hd01["demand_triggered_recommended_cards"]) >= 6

    # Verify Turn-1 `init_payload` size in `demand_triggered_zero_tax` is IDENTICAL to `Bare`!
    _, payload_bare = m6f._build_system_and_initial_brief(
        hd_tasks["hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline"],
        "Hegel_M6_9_Antithesis_LiveGPT",
        env_boot_info="linux",
        recalled_m6_notes=[],
        staged_wormhole_manifest=[],
        max_turns=18,
        cmd_timeout_sec=180,
        disclosure_mode="dedup_full",
        has_dormant_on_demand_skills=False,
    )
    _, payload_zero_tax = m6f._build_system_and_initial_brief(
        hd_tasks["hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline"],
        "Hegel_M6_9_Antithesis_LiveGPT",
        env_boot_info="linux",
        recalled_m6_notes=[{"node_id": "n1", "methodology_principle": "long text " * 200}],
        staged_wormhole_manifest=[],
        max_turns=18,
        cmd_timeout_sec=180,
        disclosure_mode="demand_triggered_zero_tax",
        has_dormant_on_demand_skills=True,
    )
    assert json.dumps(payload_zero_tax) == json.dumps(payload_bare)

    # 2. On `vr04` (`Indefinite Complex Helmholtz`), `demand_triggered_recommended_cards == []` (`0` false-positive hints!)
    _, man_vr04, diag_vr04 = m64.collect_recalled_executable_wormhole_sources(
        [],
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=v5_arch,
        task_query_text=vr_tasks["vr04__indefinite_helmholtz_wrapped_phase_inversion"].instruction_text,
        disclosure_mode="demand_triggered_zero_tax",
        staging_policy="net_value_gated",
        return_diagnostics=True,
    )
    assert man_vr04 == []
    assert diag_vr04["demand_triggered_recommended_l1_count"] == 0
    assert diag_vr04["demand_triggered_recommended_cards"] == []

    # 3. Unconfounded `M8OfflineContainerPlanner` state parity between `Bare` (`skill_activation_count=0`) and `Skill` (`skill_activation_count=5`)
    bare_st = build_container_pre_action_state(
        task_id="hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography",
        benchmark="Hegel-M91-HardDemand-2",
        wave="m91_hard_demand",
        primary_summary={
            "ended_by_finish": True,
            "turns_used": 9,
            "max_turns": 18,
            "skill_activation_count": 0,
            "nogood_count": 0,
            "staged_wormhole_modules": [],
            "final_candidate_sha256": "abc123",
        },
        contract_check={"accepted": True},
        public_metrics={"in_container_substantive_receipt_verified": True},
    )
    skill_st = build_container_pre_action_state(
        task_id="hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography",
        benchmark="Hegel-M91-HardDemand-2",
        wave="m91_hard_demand",
        primary_summary={
            "ended_by_finish": True,
            "turns_used": 9,
            "max_turns": 18,
            "skill_activation_count": 10,
            "nogood_count": 0,
            "staged_wormhole_modules": [{"module_import": "hegel_wormhole_skills.autogenous_prim_conservative_spd_solve"}],
            "final_candidate_sha256": "abc123",
        },
        contract_check={"accepted": True},
        public_metrics={"in_container_substantive_receipt_verified": True},
    )
    assert bare_st["public_verification_state"] == "SUPPORTED_IN_PARTIAL_SCOPE"
    assert skill_st["public_verification_state"] == "SUPPORTED_IN_PARTIAL_SCOPE"

    # Verify `--search` pure-JSON CLI returns top primitive without importing any skill module!
    with tempfile.TemporaryDirectory(prefix="m91_search_cli_") as td_s:
        pkg = Path(td_s) / "hegel_wormhole_skills"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text(m64.SKILL_INIT_PY, encoding="utf-8")
        (pkg / "__main__.py").write_text(m64.SKILL_MAIN_PY, encoding="utf-8")
        (pkg / "_skill_contracts.json").write_text(diag_hd01["contracts_catalog_json"], encoding="utf-8")
        for fn_m, src_m in mods_hd01.items():
            (pkg / fn_m).write_text(src_m, encoding="utf-8")
        insp_log = Path(td_s) / "insp.jsonl"
        act_log = Path(td_s) / "act.jsonl"
        env_s = {**os.environ, "PYTHONPATH": td_s, "HEGEL_SKILL_INSPECTION_LOG": str(insp_log), "HEGEL_SKILL_ACTIVATION_LOG": str(act_log)}
        pr_s = subprocess.run([sys.executable, "-m", "hegel_wormhole_skills", "--search", "preconditioner robin cg"], capture_output=True, text=True, check=True, env=env_s)
        top_line = pr_s.stdout.strip().splitlines()[0]
        assert "autogenous_prim_conservative_spd_solve" in top_line
        assert insp_log.is_file() and "skill_search_executed" in insp_log.read_text(encoding="utf-8")
        assert not act_log.exists()

    # Verify `m7_4` (`Frontier-20`) true LOTO remains `14/20`!
    m74_recs = load_historical_suite_episode_records(REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124")
    loto_passes = 0
    for i, held in enumerate(m74_recs):
        pl = M8OfflineContainerPlanner.fit_from_historical_records(m74_recs[:i] + m74_recs[i + 1 :], suite_ids=["m74_loto"])
        dec = pl.decide_second_branch(held["pre_action_state"], step_index=1, record_to_ledger=False)
        oc = held["outcomes"]
        if (oc["m8_twobranch_pass"] if dec["spawn_second_branch"] else oc["antithesis_1x_pass"]):
            loto_passes += 1
    assert loto_passes == 14


def test_hd01_and_hd02_reference_v5_composers_pass_and_naive_controls_fail() -> None:
    _, srcs = _load_v5_l1_sources()
    panel_s = srcs["autogenous_prim_reference_panel_balancing.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")
    trend_s = srcs["autogenous_prim_robust_trend_decomposition.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")
    box_s = srcs["autogenous_prim_box_ridge_bounds.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")
    spd_s = srcs["autogenous_prim_conservative_spd_solve.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")
    prof_s = srcs["autogenous_prim_ridge_profiled_matching.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")

    hd01_ref = panel_s + "\n" + trend_s + "\n" + box_s + r'''
import heapq
from scipy.optimize import lsq_linear

def _eval_primal(X, y, M, lam0, lam2, active_mask):
    p = X.shape[1]
    idx = np.flatnonzero(active_mask)
    w = np.zeros(p, dtype=float)
    if idx.size > 0:
        Xs = X[:, idx]
        w[idx] = lsq_linear(np.vstack([Xs, math.sqrt(lam2) * np.eye(idx.size)]), np.concatenate([y, np.zeros(idx.size)]), bounds=(-M, M), tol=1e-12).x
    sup = np.abs(w) > 1e-8
    w = np.where(sup, w, 0.0)
    rr = X @ w - y
    return 0.5 * float(rr @ rr) + 0.5 * lam2 * float(w @ w) + lam0 * int(np.sum(sup)), w

def _solve_block(Xb, yb, M, lam0, lam2):
    nb, pb = Xb.shape
    db = np.asarray(gram_diagonal_minorant(Xb), dtype=float)
    best_obj, best_w = _eval_primal(Xb, yb, M, lam0, lam2, np.zeros(pb, dtype=bool))
    fz0, fa0 = np.zeros(pb, dtype=bool), np.zeros(pb, dtype=bool)
    def _nb(fz, fa):
        keep = np.flatnonzero(~fz)
        cf = np.zeros(pb, dtype=float)
        if keep.size > 0:
            cf[keep] = lsq_linear(np.vstack([Xb[:, keep], math.sqrt(lam2) * np.eye(keep.size)]), np.concatenate([yb, np.zeros(keep.size)]), bounds=(-M, M), tol=1e-11).x
        _, ca = _eval_primal(Xb, yb, M, lam0, lam2, fa)
        lbf = box_ridge_cube_lower_bound(Xb, yb, cf, M, lam0, ridge=lam2, fixed_zero=fz, fixed_active=fa, diagonal=db)
        lba = box_ridge_cube_lower_bound(Xb, yb, ca, M, lam0, ridge=lam2, fixed_zero=fz, fixed_active=fa, diagonal=db)
        return (lba, ca, cf) if lba >= lbf else (lbf, cf, cf)
    lb0, c0, cf0 = _nb(fz0, fa0)
    oh, wh = _eval_primal(Xb, yb, M, lam0, lam2, np.abs(cf0) > 0.2)
    if oh < best_obj: best_obj, best_w = oh, wh
    pq = [(lb0, 0, fz0, fa0, c0, cf0)]; uid = 1; lvs = []
    while pq:
        lb, _, fz, fa, cw, cf = heapq.heappop(pq)
        if lb >= best_obj - 2e-7 or np.all(fz | fa):
            lvs.append((fz, fa, lb, cw)); continue
        free = np.flatnonzero(~(fz | fa))
        jb = int(free[int(np.argmax(np.abs(cf[free])))])
        for act in (False, True):
            fzn, fan = fz.copy(), fa.copy()
            if act: fan[jb] = True
            else: fzn[jb] = True
            o1, w1 = _eval_primal(Xb, yb, M, lam0, lam2, fan)
            if o1 < best_obj: best_obj, best_w = o1, w1
            lbn, cn, cfn = _nb(fzn, fan)
            o2, w2 = _eval_primal(Xb, yb, M, lam0, lam2, fan | ((~(fzn | fan)) & (np.abs(cfn) > 0.22)))
            if o2 < best_obj: best_obj, best_w = o2, w2
            if np.all(fzn | fan): lvs.append((fzn, fan, lbn, cn))
            else: heapq.heappush(pq, (lbn, uid, fzn, fan, cn, cfn)); uid += 1
    return best_obj, best_w, db, lvs

def solve_multicenter_certified_l0(study: dict) -> dict:
    coh = study["cohort"]
    Z_al = align_reference_panel(coh["X_raw"], coh["controls"], coh["center_id"])
    w_c = balanced_cell_weights(coh["center_id"], coh["stratum"])
    col_m = weighted_column_fill(Z_al, w_c)
    Z_imp = np.where(np.isfinite(Z_al), Z_al, col_m[None, :])
    s = np.asarray(coh["stratum"], dtype=int)
    des = np.column_stack([np.ones(len(s)), (s == 1).astype(float), (s == 2).astype(float), Z_imp])
    beta_hat, _ = cauchy_linear_fit(des, coh["y_potency"])

    sp = study["sparse_selection"]
    X, y = np.asarray(sp["X"], dtype=float), np.asarray(sp["y"], dtype=float).ravel()
    lam2, lam0, M = float(sp["lam2"]), float(sp["lam0"]), float(sp["M"])
    n, p = X.shape
    blocks, uncov = row_incidence_blocks(X)
    uc = 0.5 * float(y[uncov] @ y[uncov]) if uncov.size > 0 else 0.0
    wt, dt = np.zeros(p, dtype=float), np.zeros(p, dtype=float)
    ot = uc; bsols = []
    for ri, ci in blocks:
        bo, bw, bd, bl = _solve_block(X[np.ix_(ri, ci)], y[ri], M, lam0, lam2)
        wt[ci] = bw; dt[ci] = bd; ot += bo; bsols.append((ci, bl))
    prod = [(np.zeros(p, dtype=bool), np.zeros(p, dtype=bool), uc, np.zeros(p, dtype=float))]
    for ci, bl in bsols:
        nxt = []
        for fzg, fag, lbg, cg in prod:
            for fzb, fab, lbb, cb in bl:
                fzn, fan, cn = fzg.copy(), fag.copy(), cg.copy()
                fzn[ci] = fzb; fan[ci] = fab; cn[ci] = cb
                nxt.append((fzn, fan, lbg + lbb, cn))
        prod = nxt
    out_lvs = [{"fixed_zero": fzn.tolist(), "fixed_active": fan.tolist(), "lb": float(lbn), "minorant_diagonal": dt.tolist(), "minorant_center": cn.tolist()} for fzn, fan, lbn, cn in prod]
    clb = min(float(min(x["lb"] for x in out_lvs)), float(ot))
    return {
        "aligned_biomarkers": Z_imp,
        "robust_potency_weights": beta_hat[3:],
        "certified_selection": {"weights": wt.tolist(), "objective": float(ot), "certified_lower_bound": clb, "subcube_leaves": out_lvs},
    }
'''
    hd01_naive_ols = hd01_ref.replace("beta_hat, _ = cauchy_linear_fit(des, coh[\"y_potency\"])", "beta_hat = np.linalg.lstsq(des, coh[\"y_potency\"], rcond=None)[0]")

    hd02_ref = spd_s + "\n" + prof_s + r'''
def _assemble_3d_robin(Lx, Ly, Lz, kappa, sigma, gamma_robin):
    nx, ny, nz = kappa.shape
    steps = (Lx / nx, Ly / ny, Lz / nz)
    vol = steps[0] * steps[1] * steps[2]
    N = kappa.size
    ids = np.arange(N).reshape(nx, ny, nz)
    diag = vol * sigma.ravel().astype(float)
    rows, cols, data = [], [], []
    for axis, h_a in enumerate(steps):
        area_a = vol / h_a
        i0 = np.take(ids, np.arange(kappa.shape[axis] - 1), axis=axis).ravel()
        i1 = np.take(ids, np.arange(1, kappa.shape[axis]), axis=axis).ravel()
        k0 = kappa.ravel()[i0]; k1 = kappa.ravel()[i1]
        w = (area_a / h_a) * (2.0 * k0 * k1 / (k0 + k1))
        diag[i0] += w; diag[i1] += w
        rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-w, -w])
        for end in (0, -1):
            ib = np.take(ids, end, axis=axis).ravel()
            kb = kappa.ravel()[ib]
            diag[ib] += area_a * (gamma_robin * kb) / (gamma_robin * 0.5 * h_a + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)), vol

def solve_3d_robin_tomography(problem: dict) -> dict:
    kappa = np.asarray(problem["kappa"], dtype=float)
    sigma = np.asarray(problem["sigma"], dtype=float)
    nx, ny, nz = kappa.shape
    A, vol = _assemble_3d_robin(float(problem["Lx"]), float(problem["Ly"]), float(problem["Lz"]), kappa, sigma, float(problem["gamma_robin"]))
    M_prec = galerkin_preconditioner(A)
    cand = np.asarray(problem["candidate_sources"], dtype=float)
    M = cand.shape[0]
    bf = np.zeros_like(cand)
    for m in range(M):
        sol_m, _ = solve_residual_checked_cg(A, vol * cand[m].ravel(), preconditioner=M_prec, tol=1e-9)
        bf[m] = sol_m.reshape(nx, ny, nz)
    s_ijk = np.asarray(problem["sensor_ijk"], dtype=int)
    S = bf[:, s_ijk[:, 0], s_ijk[:, 1], s_ijk[:, 2]].T
    Z = np.asarray(problem["nuisance_design"], dtype=float)
    Y = np.asarray(problem["Y_obs"], dtype=float)
    lam_z, lam_s = float(problem["nuisance_ridge"]), float(problem["source_ridge"])
    _, Ry, whitener = fit_ridge_nuisance(Z, Y, ridge=lam_z)
    prof = np.column_stack([profile_candidate_amplitudes(S.T @ Ry[:, q], np.sum(S * S, axis=0), Z.T @ S, whitener)[0] for q in range(Y.shape[1])])
    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(Z.shape[1]), Z.T @ S], [S.T @ Z, S.T @ S + lam_s * np.eye(M)]])
    jw = np.linalg.solve(H_aug, np.vstack([Z.T @ Y, S.T @ Y]))[Z.shape[1]:, :]
    return {"basis_fields": bf, "single_source_profiled_amplitudes": prof, "joint_source_weights": jw}
'''
    hd02_raw_cg_bad = hd02_ref.replace(
        "sol_m, _ = solve_residual_checked_cg(A, vol * cand[m].ravel(), preconditioner=M_prec, tol=1e-9)",
        "sol_m, _ = cg(A, vol * cand[m].ravel(), rtol=1e-9, maxiter=450)",
    )

    with tempfile.TemporaryDirectory(prefix="m91_hd_ut_") as td:
        td_p = Path(td)
        t1_ok = _bundle_single_py_tar(td_p, "holdout_task/multicenter_certified_l0.py", hd01_ref, "hd01_ok.tar")
        r1_ok = run_hard_demand_private_verifier_by_id("hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline", t1_ok)
        assert r1_ok["strict_pass"] is True, f"HD01 ref failed: {r1_ok}"

        t1_bad = _bundle_single_py_tar(td_p, "holdout_task/multicenter_certified_l0.py", hd01_naive_ols, "hd01_bad.tar")
        r1_bad = run_hard_demand_private_verifier_by_id("hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline", t1_bad)
        assert r1_bad["strict_pass"] is False, f"HD01 naive OLS must fail: {r1_bad}"

        t2_ok = _bundle_single_py_tar(td_p, "holdout_task/robin3d_amg_tomography.py", hd02_ref, "hd02_ok.tar")
        r2_ok = run_hard_demand_private_verifier_by_id("hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography", t2_ok)
        assert r2_ok["strict_pass"] is True, f"HD02 ref failed: {r2_ok}"

        t2_bad = _bundle_single_py_tar(td_p, "holdout_task/robin3d_amg_tomography.py", hd02_raw_cg_bad, "hd02_bad.tar")
        r2_bad = run_hard_demand_private_verifier_by_id("hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography", t2_bad)
        assert r2_bad["strict_pass"] is False, f"HD02 raw unpreconditioned CG must fail: {r2_bad}"
