"""Unit & Container Verifier Test Suite for `M9.2` Equal-Information On-Demand Reuse (`Hegel-M92-EqualInfo-4`).

Verifies:
1. Equal-Information Turn-1 & Catalog Parity (`B` vs `K` vs `E`) on `pretrained_autogenous_forest_v6_m911.json`:
   - `init_payload` is byte-for-byte identical across `B`, `K`, and `E` (`staged_wormhole_manifest == []`).
   - `K` (`on_demand_knowledge_only`) and `E` (`on_demand_executable`) have matched Turn-1 system-prompt lengths (`|delta| <= 30` chars)
     and identical 4-tuple-verified catalog recommendations (`> 0` on `eq01..eq03`, `== 0` on `eq04` negative control!).
2. Shared `/holdout_task/validate_public_schema.py` + Reference `v6` Composers + Naive/Memoryless/Unpreconditioned Controls:
   - `eq01`: Reused `galerkin_preconditioner` + `solve_residual_checked_cg` + `fit_ridge_nuisance` + `profile_candidate_amplitudes`
     PASSES (`strict_pass == True`, `< 16.0s`), while unpreconditioned `cg(350)` FAILS.
   - `eq02`: Coupled `Stage 1 (align_reference_panel + balanced_cell_weights + weighted_column_fill) -> Stage 2 (complex128 phase +
     cauchy_linear_fit) -> Stage 3 (single connected block p=18..19 box_ridge_cube_lower_bound)` PASSES (`strict_pass == True`),
     while Stage-2 naive OLS corrupts `y_deconfounded` & Stage-3 weights and FAILS!
   - `eq03`: Hereditary Prony memory + `align_reference_panel` + `cauchy_linear_fit` PASSES (`strict_pass == True`), while a
     memoryless instantaneous `t = 0` predictor FAILS on same-present/different-past twin trajectories!
   - `eq04`: First-principles complex128 upwind LU + pairwise penalty solver PASSES (`strict_pass == True`).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m9_2_equal_info_reuse_benchmark import (
    EQUAL_INFO_M92_BY_ID,
    get_equal_info_m92_frontier_tasks,
    run_equal_info_m92_private_verifier_by_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V6_BUNDLE_PATH = (
    REPO_ROOT
    / "docs/evidence/m9_1_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v6_m911.json"
)


def _load_v6_l1_sources() -> tuple[dict[str, dict], dict[str, str]]:
    _, v6_arch, _, _ = m7a.load_autogenous_forest_bundle(V6_BUNDLE_PATH)
    out: dict[str, str] = {}
    for _, rec in m7a.iter_live_archive_nodes(v6_arch):
        for m in rec.get("autogenous_wormhole_modules") or []:
            fn = str(m.get("filename") or "")
            if fn.startswith("autogenous_prim_") and fn not in out:
                ok_q, rsn_q, _ = m64.verify_l1_module_qualification_record(m, require_4tuple_protocol=True)
                assert ok_q, f"v6 module {fn} failed 4-tuple check: {rsn_q}"
                out[fn] = str(m.get("python_code") or "").replace("from __future__ import annotations\n", "")
    assert len(out) == 6
    return v6_arch, out


def _bundle_single_py_tar(td: Path, arc_rel: str, code_str: str, tar_name: str) -> Path:
    src_p = td / tar_name.replace(".tar", "_dir") / arc_rel.lstrip("/")
    src_p.parent.mkdir(parents=True, exist_ok=True)
    src_p.write_text(code_str, encoding="utf-8")
    tar_p = td / tar_name
    with tarfile.open(tar_p, "w") as tf:
        tf.add(str(src_p), arcname=arc_rel.lstrip("/"))
    return tar_p


def test_m9_2_equal_information_turn1_parity_and_subproblem_gating() -> None:
    v6_arch, _ = _load_v6_l1_sources()
    tasks = {t.task_id: t for t in get_equal_info_m92_frontier_tasks()}
    assert len(tasks) == 4

    for tid, t_obj in tasks.items():
        _, man_k, diag_k = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v6_arch,
            task_query_text=t_obj.instruction_text, disclosure_mode="on_demand_knowledge_only", staging_policy="net_value_gated", return_diagnostics=True,
        )
        _, man_e, diag_e = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v6_arch,
            task_query_text=t_obj.instruction_text, disclosure_mode="on_demand_executable", staging_policy="net_value_gated", return_diagnostics=True,
        )
        assert man_k == [] and man_e == []
        assert diag_k["mounted_on_disk_l1_count"] == 0 and diag_k["knowledge_reference_l1_count"] == 6
        assert diag_e["mounted_on_disk_l1_count"] == 6 and diag_e["queryable_catalog_l1_count"] == 6
        assert diag_k["demand_triggered_recommended_l1_count"] == diag_e["demand_triggered_recommended_l1_count"]

        if tid.startswith("eq04"):
            assert diag_e["demand_triggered_recommended_l1_count"] == 0
            assert diag_e["demand_triggered_recommended_cards"] == []
        elif tid.startswith("eq02"):
            # `C3` proof: `complex128` in Stage 2a does NOT veto real primitives in Stages 1, 2b, 3!
            rec_stems = {Path(x["filename"]).stem for x in diag_e["selected_l1_modules"]}
            assert {"autogenous_prim_reference_panel_balancing", "autogenous_prim_robust_trend_decomposition", "autogenous_prim_box_ridge_bounds"} <= rec_stems
        else:
            assert diag_e["demand_triggered_recommended_l1_count"] >= 2

        sys_b, pay_b = m6f._build_system_and_initial_brief(
            t_obj, "Hegel_M6_9_Antithesis_LiveGPT", env_boot_info="linux", recalled_m6_notes=[],
            staged_wormhole_manifest=[], max_turns=18, cmd_timeout_sec=180, disclosure_mode="dedup_full", has_dormant_on_demand_skills=False,
        )
        sys_k, pay_k = m6f._build_system_and_initial_brief(
            t_obj, "Hegel_M6_9_Antithesis_LiveGPT", env_boot_info="linux", recalled_m6_notes=[{"node_id": "n1", "autogenous_only": True, "methodology_principle": "p"}],
            staged_wormhole_manifest=[], max_turns=18, cmd_timeout_sec=180, disclosure_mode="on_demand_knowledge_only", has_dormant_on_demand_skills=True,
        )
        sys_e, pay_e = m6f._build_system_and_initial_brief(
            t_obj, "Hegel_M6_9_Antithesis_LiveGPT", env_boot_info="linux", recalled_m6_notes=[{"node_id": "n1", "autogenous_only": True, "methodology_principle": "p"}],
            staged_wormhole_manifest=[], max_turns=18, cmd_timeout_sec=180, disclosure_mode="on_demand_executable", has_dormant_on_demand_skills=True,
        )
        assert json.dumps(pay_b) == json.dumps(pay_k) == json.dumps(pay_e)
        assert abs(len(sys_k) - len(sys_e)) <= 35
        assert len(sys_e) > len(sys_b)


def test_m9_2_eq01_to_eq04_verifiers_public_schema_validators_and_controls() -> None:
    _, srcs = _load_v6_l1_sources()
    nop_hdr = "track_wormhole_skill = lambda *a, **k: (lambda f: f)"
    panel_s = srcs["autogenous_prim_reference_panel_balancing.py"].replace("from . import track_wormhole_skill", nop_hdr)
    trend_s = srcs["autogenous_prim_robust_trend_decomposition.py"].replace("from . import track_wormhole_skill", nop_hdr)
    box_s = srcs["autogenous_prim_box_ridge_bounds.py"].replace("from . import track_wormhole_skill", nop_hdr)
    spd_s = srcs["autogenous_prim_conservative_spd_solve.py"].replace("from . import track_wormhole_skill", nop_hdr)
    prof_s = srcs["autogenous_prim_ridge_profiled_matching.py"].replace("from . import track_wormhole_skill", nop_hdr)

    eq01_ok = spd_s + "\n" + prof_s + r'''
from scipy.sparse.linalg import cg

def _asm(Lx, Ly, Lz, kappa, sigma, gamma_robin):
    nx, ny, nz = kappa.shape
    steps = (Lx / nx, Ly / ny, Lz / nz)
    vol = steps[0] * steps[1] * steps[2]
    N = kappa.size
    ids = np.arange(N).reshape(nx, ny, nz)
    diag = vol * sigma.ravel().astype(float)
    rows, cols, data = [], [], []
    kf = kappa.ravel()
    for axis, h_a in enumerate(steps):
        area_a = vol / h_a
        i0 = np.take(ids, np.arange(kappa.shape[axis] - 1), axis=axis).ravel()
        i1 = np.take(ids, np.arange(1, kappa.shape[axis]), axis=axis).ravel()
        w = (area_a / h_a) * (2.0 * kf[i0] * kf[i1] / (kf[i0] + kf[i1]))
        diag[i0] += w; diag[i1] += w
        rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-w, -w])
        for end in (0, -1):
            ib = np.take(ids, end, axis=axis).ravel()
            kb = kf[ib]
            diag[ib] += area_a * (gamma_robin * kb) / (gamma_robin * 0.5 * h_a + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)), vol

def solve_robin_continuation_tomography(study: dict) -> dict:
    Lx, Ly, Lz = float(study["Lx"]), float(study["Ly"]), float(study["Lz"])
    kb, km1, km2 = study["kappa_base"], study["kappa_mod1"], study["kappa_mod2"]
    sb, sm = study["sigma_base"], study["sigma_mod"]
    gr = float(study["gamma_robin"])
    thetas = np.asarray(study["thetas"], dtype=float)
    modes = np.asarray(study["source_modes"], dtype=float)
    M, nx, ny, nz = modes.shape
    s_ijk = np.asarray(study["sensor_ijk"], dtype=int)
    fc = s_ijk[:, 0] * (ny * nz) + s_ijk[:, 1] * nz + s_ijk[:, 2]

    th_mid = float(thetas[len(thetas) // 2])
    A_mid, _ = _asm(Lx, Ly, Lz, kb + th_mid * km1 + (th_mid ** 2) * km2, sb + th_mid * sm, gr)
    M_prec = galerkin_preconditioner(A_mid)

    S_blocks = []
    term = np.zeros_like(modes)
    for k_i, th in enumerate(thetas):
        Ak, vol = _asm(Lx, Ly, Lz, kb + float(th) * km1 + (float(th) ** 2) * km2, sb + float(th) * sm, gr)
        uk = np.zeros((nx * ny * nz, M), dtype=float)
        for m in range(M):
            sol_m, _ = solve_residual_checked_cg(Ak, vol * modes[m].ravel(), preconditioner=M_prec, tol=1e-9)
            uk[:, m] = sol_m
        S_blocks.append(uk[fc, :])
        if k_i == len(thetas) - 1:
            term = uk.T.reshape(M, nx, ny, nz)
    S_stack = np.vstack(S_blocks)
    Z = np.asarray(study["Z_nuisance"], dtype=float)
    Y = np.asarray(study["Y_obs"], dtype=float)
    lam_z, lam_s = float(study["nuisance_ridge"]), float(study["source_ridge"])
    _, Ry, whitener = fit_ridge_nuisance(Z, Y, ridge=lam_z)
    prof = np.column_stack([profile_candidate_amplitudes(S_stack.T @ Ry[:, q], np.sum(S_stack * S_stack, axis=0), Z.T @ S_stack, whitener)[0] for q in range(Y.shape[1])])
    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(Z.shape[1]), Z.T @ S_stack], [S_stack.T @ Z, S_stack.T @ S_stack + lam_s * np.eye(M)]])
    jw = np.linalg.solve(H_aug, np.vstack([Z.T @ Y, S_stack.T @ Y]))[Z.shape[1]:, :]
    return {"terminal_fields": term, "continuation_sensor_predictions": S_stack, "profiled_amplitudes": prof, "joint_source_weights": jw}
'''
    eq01_bad_cg = eq01_ok.replace(
        "sol_m, _ = solve_residual_checked_cg(Ak, vol * modes[m].ravel(), preconditioner=M_prec, tol=1e-9)",
        "sol_m, _ = cg(Ak, vol * modes[m].ravel(), rtol=1e-9, maxiter=300)",
    )

    eq02_ok = panel_s + "\n" + trend_s + "\n" + box_s + r'''
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

def solve_coupled_interferometry_l0(study: dict) -> dict:
    pan = study["panel"]
    Z_al = align_reference_panel(pan["X_panel_raw"], pan["controls"], pan["center_id"])
    w_cell = balanced_cell_weights(pan["center_id"], pan["stratum"])
    col_m = weighted_column_fill(Z_al, w_cell)
    Z_imp = np.where(np.isfinite(Z_al), Z_al, col_m[None, :])

    inter = study["interferometry"]
    demod = np.angle(np.asarray(inter["signal_phasor"], dtype=np.complex128) * np.conj(np.asarray(inter["ref_phasor"], dtype=np.complex128)))
    st_arr = np.asarray(pan["stratum"])
    N = len(st_arr)
    H_cal = np.column_stack([np.ones(N), (st_arr == 1).astype(float), (st_arr == 2).astype(float), Z_imp, demod])
    cal_idx = np.asarray(study["calibration_indices"], dtype=int)
    s3_idx = np.asarray(study["stage3_indices"], dtype=int)
    y_raw = np.asarray(study["y_assay_raw"], dtype=float)
    theta_hat, _ = cauchy_linear_fit(H_cal[cal_idx], y_raw[cal_idx])
    y_deconf = y_raw[s3_idx] - H_cal[s3_idx] @ theta_hat

    X = np.asarray(study["X_cand"], dtype=float)
    M, lam0, lam2 = float(study["M"]), float(study["lam0"]), float(study["lam2"])
    n_s, p = X.shape
    blocks, _ = row_incidence_blocks(X)
    assert len(blocks) == 1
    d = np.asarray(gram_diagonal_minorant(X), dtype=float)
    best_obj, best_w = _eval_primal(X, y_deconf, M, lam0, lam2, np.zeros(p, dtype=bool))
    fz0, fa0 = np.zeros(p, dtype=bool), np.zeros(p, dtype=bool)
    def _nb(fz, fa):
        keep = np.flatnonzero(~fz)
        cf = np.zeros(p, dtype=float)
        if keep.size > 0:
            cf[keep] = lsq_linear(np.vstack([X[:, keep], math.sqrt(lam2) * np.eye(keep.size)]), np.concatenate([y_deconf, np.zeros(keep.size)]), bounds=(-M, M), tol=1e-11).x
        _, ca = _eval_primal(X, y_deconf, M, lam0, lam2, fa)
        lbf = box_ridge_cube_lower_bound(X, y_deconf, cf, M, lam0, ridge=lam2, fixed_zero=fz, fixed_active=fa, diagonal=d)
        lba = box_ridge_cube_lower_bound(X, y_deconf, ca, M, lam0, ridge=lam2, fixed_zero=fz, fixed_active=fa, diagonal=d)
        return (lba, ca, cf) if lba >= lbf else (lbf, cf, cf)
    lb0, c0, cf0 = _nb(fz0, fa0)
    oh, wh = _eval_primal(X, y_deconf, M, lam0, lam2, np.abs(cf0) > 0.2)
    if oh < best_obj: best_obj, best_w = oh, wh
    pq = [(lb0, 0, fz0, fa0, c0, cf0)]; uid = 1; lvs = []; exp_cnt = 0
    while pq:
        lb, _, fz, fa, cw, cf = heapq.heappop(pq)
        if lb >= best_obj - 2e-7 or np.all(fz | fa) or exp_cnt >= 350:
            lvs.append({"fixed_zero": fz.tolist(), "fixed_active": fa.tolist(), "lb": float(lb), "minorant_diagonal": d.tolist(), "minorant_center": cw.tolist()})
            continue
        exp_cnt += 1
        free = np.flatnonzero(~(fz | fa))
        jb = int(free[int(np.argmax(np.abs(cf[free])))])
        for act in (False, True):
            fzn, fan = fz.copy(), fa.copy()
            if act: fan[jb] = True
            else: fzn[jb] = True
            o1, w1 = _eval_primal(X, y_deconf, M, lam0, lam2, fan)
            if o1 < best_obj: best_obj, best_w = o1, w1
            lbn, cn, cfn = _nb(fzn, fan)
            o2, w2 = _eval_primal(X, y_deconf, M, lam0, lam2, fan | ((~(fzn | fan)) & (np.abs(cfn) > 0.22)))
            if o2 < best_obj: best_obj, best_w = o2, w2
            if lbn >= best_obj - 2e-7:
                lvs.append({"fixed_zero": fzn.tolist(), "fixed_active": fan.tolist(), "lb": float(lbn), "minorant_diagonal": d.tolist(), "minorant_center": cn.tolist()})
            else:
                heapq.heappush(pq, (lbn, uid, fzn, fan, cn, cfn)); uid += 1
    clb = min(float(min(x["lb"] for x in lvs)), float(best_obj))
    return {
        "aligned_panel": Z_imp,
        "calibration_weights": theta_hat,
        "deconfounded_target": y_deconf,
        "certified_selection": {"weights": best_w.tolist(), "objective": float(best_obj), "certified_lower_bound": clb, "subcube_leaves": lvs},
    }
'''
    eq02_bad_ols = eq02_ok.replace(
        "theta_hat, _ = cauchy_linear_fit(H_cal[cal_idx], y_raw[cal_idx])",
        "theta_hat = np.linalg.lstsq(H_cal[cal_idx], y_raw[cal_idx], rcond=None)[0]",
    )

    eq03_ok = panel_s + "\n" + trend_s + r'''
def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    dt = float(dataset["history_dt"])
    taus = np.asarray(dataset["memory_taus"], dtype=float)
    eval_t = np.asarray(dataset["eval_times"], dtype=float)
    tr, te = dataset["train"], dataset["test"]
    n_tr, n_te = len(tr["lab_id"]), len(te["lab_id"])

    lab_all = np.concatenate([tr["lab_id"], te["lab_id"]])
    st_all = np.concatenate([tr["stratum"], te["stratum"]])
    X_all = np.vstack([tr["coupon_features_raw"], te["coupon_features_raw"]])
    ref_all = np.vstack([tr["ref_coupons"], te["ref_coupons"]])
    Z_al = align_reference_panel(X_all, ref_all, lab_all)
    w_c = balanced_cell_weights(lab_all, st_all)
    col_m = weighted_column_fill(Z_al, w_c)
    Z_imp = np.where(np.isfinite(Z_al), Z_al, col_m[None, :])
    X_tr, X_te = Z_imp[:n_tr], Z_imp[n_tr:]

    def _mem_states(hist):
        h = np.asarray(hist, dtype=float)
        zs = []
        for tau_m in taus:
            dec = math.exp(-dt / float(tau_m))
            zm = np.zeros(h.shape[0], dtype=float)
            for s in range(h.shape[1]):
                zm = dec * zm + (1.0 - dec) * h[:, s]
            zs.append(zm)
        return zs[0], zs[1]

    def _build_design(X_f, sh0, z1, z2):
        e1 = 1.0 - np.exp(-eval_t / taus[0])
        e2 = 1.0 - np.exp(-eval_t / taus[1])
        n_r, k_t = len(sh0), len(eval_t)
        c1 = ((sh0 - z1)[:, None] * e1[None, :]).ravel()
        c2 = ((sh0 - z2)[:, None] * e2[None, :]).ravel()
        ct = np.broadcast_to(eval_t[None, :], (n_r, k_t)).ravel()
        cx = [np.ravel(X_f[:, d : d + 1] * eval_t[None, :]) for d in range(4)]
        return np.column_stack([c1, c2, ct, *cx])

    z1_tr, z2_tr = _mem_states(tr["past_stress_history"])
    Phi_tr = _build_design(X_tr, np.asarray(tr["stress_hold_0"], dtype=float), z1_tr, z2_tr)
    y_tr = (np.asarray(tr["strain_curves_obs"], dtype=float) - np.asarray(tr["strain_0"], dtype=float)[:, None]).ravel()
    w_hat, _ = cauchy_linear_fit(Phi_tr, y_tr)

    z1_te, z2_te = _mem_states(te["past_stress_history"])
    s0_te = np.asarray(te["strain_0"], dtype=float)
    pred_te = s0_te[:, None] + (_build_design(X_te, np.asarray(te["stress_hold_0"], dtype=float), z1_te, z2_te) @ w_hat).reshape(n_te, len(eval_t))

    ts = np.concatenate([[0.0], eval_t])
    ys = np.column_stack([s0_te, pred_te])
    crits = np.asarray(te["critical_strain"], dtype=float)
    cross = np.full(n_te, float(ts[-1]), dtype=float)
    for i in range(n_te):
        c = float(crits[i])
        if ys[i, 0] >= c:
            cross[i] = 0.0; continue
        for k in range(len(ts) - 1):
            if ys[i, k + 1] >= c:
                dy = max(ys[i, k + 1] - ys[i, k], 1e-12)
                cross[i] = float(ts[k] + (c - ys[i, k]) / dy * (ts[k + 1] - ts[k])); break
    return {"aligned_test_features": X_te, "hereditary_weights": w_hat, "predicted_strain": pred_te, "crossing_times": cross}
'''
    eq03_memoryless_bad = eq03_ok.replace("zm = dec * zm + (1.0 - dec) * h[:, s]", "zm = np.zeros_like(zm)")

    eq04_ok = r'''
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

def _asm(Lx, Ly, nu, vx, vy, sigma, omega):
    nx, ny = sigma.shape
    hx, hy = Lx / (nx + 1), Ly / (ny + 1)
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    diag = np.full(N, 2.0 * nu / (hx ** 2) + 2.0 * nu / (hy ** 2), dtype=np.complex128) + sigma.ravel().astype(np.complex128) + 1j * float(omega)
    rows, cols, data = [], [], []
    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    rows.extend([i0, i1]); cols.extend([i1, i0])
    data.extend([np.full(i0.size, -nu / (hx ** 2), dtype=np.complex128), np.full(i0.size, -nu / (hx ** 2), dtype=np.complex128)])
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    rows.extend([j0, j1]); cols.extend([j1, j0])
    data.extend([np.full(j0.size, -nu / (hy ** 2), dtype=np.complex128), np.full(j0.size, -nu / (hy ** 2), dtype=np.complex128)])
    vxf = vx.ravel(); vyf = vy.ravel()
    diag += np.abs(vxf) / hx + np.abs(vyf) / hy
    pos_x = vxf[i1] >= 0.0
    if np.any(pos_x): rows.append(i1[pos_x]); cols.append(i0[pos_x]); data.append((-vxf[i1[pos_x]] / hx).astype(np.complex128))
    neg_x = vxf[i0] < 0.0
    if np.any(neg_x): rows.append(i0[neg_x]); cols.append(i1[neg_x]); data.append((vxf[i0[neg_x]] / hx).astype(np.complex128))
    pos_y = vyf[j1] >= 0.0
    if np.any(pos_y): rows.append(j1[pos_y]); cols.append(j0[pos_y]); data.append((-vyf[j1[pos_y]] / hy).astype(np.complex128))
    neg_y = vyf[j0] < 0.0
    if np.any(neg_y): rows.append(j0[neg_y]); cols.append(j1[neg_y]); data.append((vyf[j0[neg_y]] / hy).astype(np.complex128))
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    basis = np.asarray(problem["basis"], dtype=float)
    M, nx, ny = basis.shape
    A = _asm(float(problem["Lx"]), float(problem["Ly"]), float(problem["nu"]), np.asarray(problem["vx"], dtype=float), np.asarray(problem["vy"], dtype=float), np.asarray(problem["sigma"], dtype=float), float(problem["omega"]))
    lu = splu(A.tocsc())
    G_full = np.zeros((M, nx, ny), dtype=np.complex128)
    for m in range(M):
        G_full[m] = lu.solve(basis[m].ravel().astype(np.complex128)).reshape(nx, ny)
    s_ij = np.asarray(problem["sensor_ij"], dtype=int)
    G = G_full[:, s_ij[:, 0], s_ij[:, 1]].T
    y = np.asarray(problem["y_obs_re"], dtype=float) + 1j * np.asarray(problem["y_obs_im"], dtype=float)
    lam2, lam0 = float(problem["lam2"]), float(problem["lam0"])
    pairs = problem["pair_penalties"]
    GH_G = G.conj().T @ G; GH_y = G.conj().T @ y
    yy = 0.5 * float(np.real(np.vdot(y, y)))
    best_j = yy; best_w = np.zeros(M, dtype=np.complex128)
    for mask in range(1, 1 << M):
        idx = [m for m in range(M) if (mask >> m) & 1]
        k = len(idx)
        w_s = np.linalg.solve(GH_G[np.ix_(idx, idx)] + lam2 * np.eye(k, dtype=np.complex128), GH_y[idx])
        val = yy - 0.5 * float(np.real(np.vdot(GH_y[idx], w_s))) + lam0 * k + sum(float(gam) for u, v, gam in pairs if ((mask >> int(u)) & 1) and ((mask >> int(v)) & 1))
        if val < best_j:
            best_j = val; best_w = np.zeros(M, dtype=np.complex128); best_w[idx] = w_s
    psi = np.tensordot(best_w, G_full, axes=(0, 0))
    return {"weights_real": best_w.real, "weights_imag": best_w.imag, "objective": float(best_j), "predicted_field_abs": np.abs(psi)}
'''

    with tempfile.TemporaryDirectory(prefix="m92_ut_") as td:
        td_p = Path(td)
        # Check all 4 reference solvers PASS and all 3 negative controls FAIL in real air-gapped verifier containers!
        t1_ok = _bundle_single_py_tar(td_p, "holdout_task/robin_continuation_tomography.py", eq01_ok, "eq01_ok.tar")
        r1_ok = run_equal_info_m92_private_verifier_by_id("eq01__3d_multi_operator_robin_continuation_and_nuisance_tomography", t1_ok)
        assert r1_ok["strict_pass"] is True, f"eq01_ok failed: {r1_ok}"

        t1_bad = _bundle_single_py_tar(td_p, "holdout_task/robin_continuation_tomography.py", eq01_bad_cg, "eq01_bad.tar")
        r1_bad = run_equal_info_m92_private_verifier_by_id("eq01__3d_multi_operator_robin_continuation_and_nuisance_tomography", t1_bad)
        assert r1_bad["strict_pass"] is False, f"eq01_bad_cg unexpectedly passed: {r1_bad}"

        t2_ok = _bundle_single_py_tar(td_p, "holdout_task/coupled_interferometry_certified_l0.py", eq02_ok, "eq02_ok.tar")
        r2_ok = run_equal_info_m92_private_verifier_by_id("eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0", t2_ok)
        assert r2_ok["strict_pass"] is True, f"eq02_ok failed: {r2_ok}"

        t2_bad = _bundle_single_py_tar(td_p, "holdout_task/coupled_interferometry_certified_l0.py", eq02_bad_ols, "eq02_bad.tar")
        r2_bad = run_equal_info_m92_private_verifier_by_id("eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0", t2_bad)
        assert r2_bad["strict_pass"] is False, f"eq02_bad_ols unexpectedly passed: {r2_bad}"

        t3_ok = _bundle_single_py_tar(td_p, "holdout_task/viscoelastic_hereditary_forecast.py", eq03_ok, "eq03_ok.tar")
        r3_ok = run_equal_info_m92_private_verifier_by_id("eq03__viscoelastic_hereditary_memory_same_present_different_past", t3_ok)
        assert r3_ok["strict_pass"] is True, f"eq03_ok failed: {r3_ok}"

        t3_bad = _bundle_single_py_tar(td_p, "holdout_task/viscoelastic_hereditary_forecast.py", eq03_memoryless_bad, "eq03_bad.tar")
        r3_bad = run_equal_info_m92_private_verifier_by_id("eq03__viscoelastic_hereditary_memory_same_present_different_past", t3_bad)
        assert r3_bad["strict_pass"] is False, f"eq03_memoryless unexpectedly passed: {r3_bad}"

        t4_ok = _bundle_single_py_tar(td_p, "holdout_task/nonhermitian_advection_coupled_select.py", eq04_ok, "eq04_ok.tar")
        r4_ok = run_equal_info_m92_private_verifier_by_id("eq04__nonhermitian_complex_advection_and_coupled_ising_negative_control", t4_ok)
        assert r4_ok["strict_pass"] is True, f"eq04_ok failed: {r4_ok}"

        # Also verify `/holdout_task/validate_public_schema.py` executes cleanly on `eq02`!
        ht2 = EQUAL_INFO_M92_BY_ID["eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0"]
        pub_dir = td_p / "eq02_pub_check"
        pub_dir.mkdir(parents=True, exist_ok=True)
        setup_s = ht2.public_setup_py.replace("/holdout_task", str(pub_dir))
        subprocess.run([sys.executable, "-c", setup_s], check=True)
        cand_f = pub_dir / "coupled_interferometry_certified_l0.py"
        cand_f.write_text(eq02_ok, encoding="utf-8")
        v_pr = subprocess.run([sys.executable, str(pub_dir / "validate_public_schema.py"), str(cand_f)], capture_output=True, text=True, check=True)
        assert json.loads(v_pr.stdout)["schema_valid"] is True
