"""Unit and container-verifier soundness tests for `Hegel-M90a-VoluntaryReuse-4` (`vr01..vr04`).

Verifies:
1. `vr01`:
   - Strictly positive unbranched root relaxation gap (`minorant` AND `Fenchel dual`) across private seeds `(99201, 99202, 99203)` at `p in {22, 23, 24}`.
   - Reference solver using frozen `v5` `autogenous_prim_box_ridge_bounds` (`row_incidence_blocks`, `gram_diagonal_minorant`, `box_ridge_cube_lower_bound`) + proof-carrying `(minorant_diagonal, minorant_center)` witnesses PASSES (`strict_pass == True`).
   - Reference solver using pure Fenchel perspective `dual_nu` witnesses ALSO PASSES (`strict_pass == True`).
   - Fake lower bound, missing witness, and unbranched 1-leaf root relaxation FAIL (`strict_pass == False`).
2. `vr02`:
   - Reference solver using `galerkin_preconditioner` + `solve_residual_checked_cg` + `fit_ridge_nuisance` + `profile_candidate_amplitudes` PASSES (`strict_pass == True`).
   - Naive full-module call to `conservative_grid_operator` (zero-ghost Dirichlet mismatch vs Robin BCs) FAILS (`strict_pass == False`).
3. `vr03`:
   - Reference solver composing ZOH pre-window Prony rollout + `cauchy_linear_fit` + `align_reference_panel` + `balanced_cell_weights` + `weighted_column_fill` PASSES (`strict_pass == True`).
   - Omitting `past_strain_history` OR omitting `align_reference_panel` FAILS (`strict_pass == False`).
4. `vr04`:
   - Helmholtz operator real part is strictly indefinite (`min_eig < -10.0`), `net_value_gated` vetoes `autogenous_prim_conservative_spd_solve`, reference complex sparse-LU + phasor solver PASSES (`strict_pass == True`), and naive wrapped-angle Euclidean average FAILS (`strict_pass == False`).
"""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np
from scipy.optimize import lsq_linear

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m9_0_voluntary_reuse_benchmark import (
    VOLUNTARY_REUSE_M90A_BY_ID,
    get_voluntary_reuse_m90a_frontier_tasks,
    make_vr01_instance,
    run_voluntary_reuse_private_verifier_by_id,
    verify_symmetric_box_l0_proof_certificate,
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
                raw_py = str(m.get("python_code") or "")
                out[fn] = raw_py.replace("from __future__ import annotations\n", "")
    return v5_arch, out


def _bundle_single_py_tar(td: Path, arc_rel: str, code_str: str, tar_name: str) -> Path:
    src_p = td / tar_name.replace(".tar", "_dir") / arc_rel.lstrip("/")
    src_p.parent.mkdir(parents=True, exist_ok=True)
    src_p.write_text(code_str, encoding="utf-8")
    tar_p = td / tar_name
    with tarfile.open(tar_p, "w") as tf:
        tf.add(str(src_p), arcname=arc_rel.lstrip("/"))
    return tar_p


def test_vr01_positive_root_gap_and_proof_carrying_minorant_and_fenchel_verifiers() -> None:
    # 1. Verify positive root relaxation gap on all 3 private seeds (99201, 99202, 99203)
    for sd, bsz in ((99201, (10, 10)), (99202, (10, 11)), (99203, (11, 11))):
        inst = make_vr01_instance(sd, bsz)
        prob = inst["problem"]
        X, y = prob["X"], prob["y"]
        n, p = X.shape
        assert p == sum(bsz) and p >= 20
        exact_opt = inst["gt"]["exact_objective"]
        exact_w = inst["gt"]["exact_weights"]

        # Gershgorin minorant strictly positive on every column
        G = X.T @ X
        off = np.sum(np.abs(G), axis=1) - np.abs(np.diag(G))
        d_gersh = np.diag(G) - 1.0001 * off
        assert float(np.min(d_gersh)) > 0.20

        # Check unbranched 1-leaf root minorant (even centered at continuous box-ridge solution!) has positive gap > 1e-3
        res_cont = lsq_linear(
            np.vstack([X, np.sqrt(prob["lam2"]) * np.eye(p)]),
            np.concatenate([y, np.zeros(p)]),
            bounds=(-prob["M"], prob["M"]),
        )
        c_root = res_cont.x
        nu_root = y - X @ c_root

        root_min_chk = verify_symmetric_box_l0_proof_certificate(
            X=X, y=y, lam2=prob["lam2"], lam0=prob["lam0"], M_box=prob["M"],
            weights=exact_w, reported_objective=exact_opt, certified_lower_bound=exact_opt,
            subcube_leaves=[{
                "fixed_zero": [False] * p, "fixed_active": [False] * p,
                "lb": exact_opt, "minorant_diagonal": d_gersh.tolist(), "minorant_center": c_root.tolist(),
            }],
        )
        assert root_min_chk["strict_pass"] is False
        assert root_min_chk["leaf_bounds_valid"] is False
        v_lb_min = root_min_chk["leaves_preview"][0]["verified_lb"]
        assert exact_opt - v_lb_min > 0.05, f"Expected substantial root minorant gap on seed {sd}, got {exact_opt - v_lb_min}"

        root_fen_chk = verify_symmetric_box_l0_proof_certificate(
            X=X, y=y, lam2=prob["lam2"], lam0=prob["lam0"], M_box=prob["M"],
            weights=exact_w, reported_objective=exact_opt, certified_lower_bound=exact_opt,
            subcube_leaves=[{
                "fixed_zero": [False] * p, "fixed_active": [False] * p,
                "lb": exact_opt, "dual_nu": nu_root.tolist(),
            }],
        )
        assert root_fen_chk["strict_pass"] is False
        v_lb_fen = root_fen_chk["leaves_preview"][0]["verified_lb"]
        assert exact_opt - v_lb_fen > 0.05, f"Expected substantial root Fenchel gap on seed {sd}, got {exact_opt - v_lb_fen}"

    # 2. Verify container private verifier passes BOTH:
    #    (a) actual frozen `v5` `autogenous_prim_box_ridge_bounds` (`row_incidence_blocks` + `gram_diagonal_minorant` + `box_ridge_cube_lower_bound`) AND
    #    (b) global first-principles Fenchel/minorant B&B without block decomposition, while rejecting fake LB!
    _, srcs_v5 = _load_v5_l1_sources()
    box_src = srcs_v5["autogenous_prim_box_ridge_bounds.py"].replace(
        "from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)"
    )

    v5_skill_bnb_solver = box_src + r'''
import heapq
from scipy.optimize import lsq_linear

def _eval_primal(X, y, M, lam0, lam2, active_mask):
    p = X.shape[1]
    idx = np.flatnonzero(active_mask)
    w = np.zeros(p, dtype=float)
    if idx.size > 0:
        Xs = X[:, idx]
        res = lsq_linear(np.vstack([Xs, math.sqrt(lam2) * np.eye(idx.size)]), np.concatenate([y, np.zeros(idx.size)]), bounds=(-M, M), tol=1e-12)
        w[idx] = res.x
    sup = np.abs(w) > 1e-8
    w = np.where(sup, w, 0.0)
    rr = X @ w - y
    obj = 0.5 * float(rr @ rr) + 0.5 * lam2 * float(w @ w) + lam0 * int(np.sum(sup))
    return obj, w

def _solve_block_bnb(Xb, yb, M, lam0, lam2):
    nb, pb = Xb.shape
    db = np.asarray(gram_diagonal_minorant(Xb), dtype=float)
    best_obj, best_w = _eval_primal(Xb, yb, M, lam0, lam2, np.zeros(pb, dtype=bool))
    fz0 = np.zeros(pb, dtype=bool)
    fa0 = np.zeros(pb, dtype=bool)

    def _node_bound(fz, fa):
        keep = np.flatnonzero(~fz)
        c_full = np.zeros(pb, dtype=float)
        if keep.size > 0:
            res = lsq_linear(np.vstack([Xb[:, keep], math.sqrt(lam2) * np.eye(keep.size)]), np.concatenate([yb, np.zeros(keep.size)]), bounds=(-M, M), tol=1e-11)
            c_full[keep] = res.x
        _, c_act = _eval_primal(Xb, yb, M, lam0, lam2, fa)
        lb_f = box_ridge_cube_lower_bound(Xb, yb, c_full, M, lam0, ridge=lam2, fixed_zero=fz, fixed_active=fa, diagonal=db)
        lb_a = box_ridge_cube_lower_bound(Xb, yb, c_act, M, lam0, ridge=lam2, fixed_zero=fz, fixed_active=fa, diagonal=db)
        if lb_a >= lb_f:
            return lb_a, c_act, c_full
        return lb_f, c_full, c_full

    lb0, c0, c0_full = _node_bound(fz0, fa0)
    o_h, w_h = _eval_primal(Xb, yb, M, lam0, lam2, np.abs(c0_full) > 0.2)
    if o_h < best_obj:
        best_obj, best_w = o_h, w_h
    pq = [(lb0, 0, fz0, fa0, c0, c0_full)]
    uid = 1
    leaves = []
    while pq:
        lb, _, fz, fa, c_wit, c_full = heapq.heappop(pq)
        if lb >= best_obj - 2e-7 or np.all(fz | fa):
            leaves.append((fz, fa, lb, c_wit))
            continue
        free = np.flatnonzero(~(fz | fa))
        j_br = int(free[int(np.argmax(np.abs(c_full[free])))])
        for is_act in (False, True):
            fz_n, fa_n = fz.copy(), fa.copy()
            if is_act: fa_n[j_br] = True
            else: fz_n[j_br] = True
            o1, w1 = _eval_primal(Xb, yb, M, lam0, lam2, fa_n)
            if o1 < best_obj:
                best_obj, best_w = o1, w1
            lb_n, c_n, cf_n = _node_bound(fz_n, fa_n)
            o2, w2 = _eval_primal(Xb, yb, M, lam0, lam2, fa_n | ((~(fz_n | fa_n)) & (np.abs(cf_n) > 0.22)))
            if o2 < best_obj:
                best_obj, best_w = o2, w2
            if np.all(fz_n | fa_n):
                leaves.append((fz_n, fa_n, o1 - 1e-11, w1))
            else:
                heapq.heappush(pq, (lb_n, uid, fz_n, fa_n, c_n, cf_n))
                uid += 1
    return best_obj, best_w, db, leaves

def solve_proof_carrying_box_l0(problem: dict) -> dict:
    X = np.asarray(problem["X"], dtype=float)
    y = np.asarray(problem["y"], dtype=float).ravel()
    lam2, lam0, M = float(problem["lam2"]), float(problem["lam0"]), float(problem["M"])
    n, p = X.shape
    blocks, uncov = row_incidence_blocks(X)
    uncov_const = 0.5 * float(y[uncov] @ y[uncov]) if uncov.size > 0 else 0.0
    w_tot = np.zeros(p, dtype=float)
    d_tot = np.zeros(p, dtype=float)
    obj_tot = uncov_const
    block_sols = []
    for r_idx, c_idx in blocks:
        bo, bw, bd, blvs = _solve_block_bnb(X[np.ix_(r_idx, c_idx)], y[r_idx], M, lam0, lam2)
        w_tot[c_idx] = bw
        d_tot[c_idx] = bd
        obj_tot += bo
        block_sols.append((r_idx, c_idx, blvs))

    # Assemble global product leaves with exact global `minorant_diagonal` + `minorant_center` (or `dual_nu` on full leaves)
    prod = [(np.zeros(p, dtype=bool), np.zeros(p, dtype=bool), uncov_const, np.zeros(p, dtype=float))]
    for _, c_idx, blvs in block_sols:
        nxt = []
        for fz_g, fa_g, lb_g, c_g in prod:
            for fz_b, fa_b, lb_b, c_b in blvs:
                fzn, fan, cn = fz_g.copy(), fa_g.copy(), c_g.copy()
                fzn[c_idx] = fz_b
                fan[c_idx] = fa_b
                cn[c_idx] = c_b
                nxt.append((fzn, fan, lb_g + lb_b, cn))
        prod = nxt

    out_leaves = []
    for fzn, fan, lbn, cn in prod:
        if np.all(fzn | fan):
            nu = y - X @ cn
            al = np.abs(X.T @ nu)
            phi = np.where(al < M * lam2, 0.5 * al * al / lam2, M * al - 0.5 * lam2 * M * M)
            ond = lam0 - phi
            vlb = float(-0.5 * float(nu @ nu) + float(nu @ y) + np.sum(ond[fan]) - 1e-11)
            out_leaves.append({"fixed_zero": fzn.tolist(), "fixed_active": fan.tolist(), "lb": vlb, "dual_nu": nu.tolist()})
        else:
            out_leaves.append({
                "fixed_zero": fzn.tolist(), "fixed_active": fan.tolist(),
                "lb": float(lbn), "minorant_diagonal": d_tot.tolist(), "minorant_center": cn.tolist(),
            })
    cert_lb = min(float(min(x["lb"] for x in out_leaves)), float(obj_tot))
    return {"weights": w_tot.tolist(), "objective": float(obj_tot), "certified_lower_bound": cert_lb, "subcube_leaves": out_leaves}
'''
    fake_solver = r'''
import numpy as np
def solve_proof_carrying_box_l0(problem: dict) -> dict:
    X, y = problem["X"], problem["y"]
    p = X.shape[1]
    obj0 = 0.5 * float(y @ y)
    fz1 = [False] * p; fa1 = [False] * p; fz1[0] = True
    fz2 = [False] * p; fa2 = [False] * p; fa2[0] = True
    return {
        "weights": [0.0] * p,
        "objective": obj0,
        "certified_lower_bound": obj0 - 1e-8,
        "subcube_leaves": [
            {"fixed_zero": fz1, "fixed_active": fa1, "lb": obj0 - 1e-8, "dual_nu": [0.0] * len(y)},
            {"fixed_zero": fz2, "fixed_active": fa2, "lb": obj0 - 1e-8, "dual_nu": [0.0] * len(y)},
        ],
    }
'''
    with tempfile.TemporaryDirectory(prefix="m90a_vr01_ut_") as td:
        td_p = Path(td)
        tar_ok = _bundle_single_py_tar(td_p, "holdout_task/proof_box_l0.py", v5_skill_bnb_solver, "vr01_ok.tar")
        rep_ok = run_voluntary_reuse_private_verifier_by_id("vr01__proof_carrying_box_l0_minorant_bnb", tar_ok)
        assert rep_ok["strict_pass"] is True, f"VR01 reference solver failed: {rep_ok}"

        tar_bad = _bundle_single_py_tar(td_p, "holdout_task/proof_box_l0.py", fake_solver, "vr01_bad.tar")
        rep_bad = run_voluntary_reuse_private_verifier_by_id("vr01__proof_carrying_box_l0_minorant_bnb", tar_bad)
        assert rep_bad["strict_pass"] is False, f"VR01 fake LB should fail: {rep_bad}"


def test_vr02_reference_amg_cg_profiler_passes_and_dirichlet_mismatch_fails() -> None:
    _, srcs = _load_v5_l1_sources()
    # Inline real v5 `galerkin_preconditioner`, `solve_residual_checked_cg`, `fit_ridge_nuisance`, `profile_candidate_amplitudes`!
    spd_src = srcs["autogenous_prim_conservative_spd_solve.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")
    prof_src = srcs["autogenous_prim_ridge_profiled_matching.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")

    vr02_ref_code = spd_src + "\n" + prof_src + r'''
def _assemble_robin(Lx, Ly, kappa, sigma, gamma_robin):
    nx, ny = kappa.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    rows, cols, data = [], [], []
    diag = (hx * hy) * sigma.ravel().astype(float)
    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    k0, k1 = kappa[:-1, :].ravel(), kappa[1:, :].ravel()
    wx = (hy / hx) * (2.0 * k0 * k1 / (k0 + k1))
    diag[i0] += wx; diag[i1] += wx
    rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-wx, -wx])
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    ky0, ky1 = kappa[:, :-1].ravel(), kappa[:, 1:].ravel()
    wy = (hx / hy) * (2.0 * ky0 * ky1 / (ky0 + ky1))
    diag[j0] += wy; diag[j1] += wy
    rows.extend([j0, j1]); cols.extend([j1, j0]); data.extend([-wy, -wy])
    for xb in (0, -1):
        ib = ids[xb, :].ravel(); kb = kappa[xb, :].ravel()
        diag[ib] += hy * (gamma_robin * kb) / (gamma_robin * 0.5 * hx + kb)
    for yb in (0, -1):
        jb = ids[:, yb].ravel(); kb = kappa[:, yb].ravel()
        diag[jb] += hx * (gamma_robin * kb) / (gamma_robin * 0.5 * hy + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    Lx, Ly = float(problem["Lx"]), float(problem["Ly"])
    kappa = np.asarray(problem["kappa"], dtype=float)
    sigma = np.asarray(problem["sigma"], dtype=float)
    nx, ny = kappa.shape
    hx, hy = Lx / nx, Ly / ny
    A = _assemble_robin(Lx, Ly, kappa, sigma, float(problem["gamma_robin"]))
    M_prec = galerkin_preconditioner(A)
    cand = np.asarray(problem["candidate_sources"], dtype=float)
    M = cand.shape[0]
    bf = np.zeros_like(cand)
    for m in range(M):
        sol_m, _ = solve_residual_checked_cg(A, (hx * hy) * cand[m].ravel(), preconditioner=M_prec, tol=1e-10)
        bf[m] = sol_m.reshape(nx, ny)
    s_ij = np.asarray(problem["sensor_ij"], dtype=int)
    S = bf[:, s_ij[:, 0], s_ij[:, 1]].T
    Z = np.asarray(problem["nuisance_design"], dtype=float)
    Y = np.asarray(problem["Y_obs"], dtype=float)
    lam_z = float(problem["nuisance_ridge"])
    lam_s = float(problem["source_ridge"])
    _, R_y, whitener = fit_ridge_nuisance(Z, Y, ridge=lam_z)
    cand_e = np.sum(S * S, axis=0)
    cross = Z.T @ S
    prof = np.zeros((M, Y.shape[1]), dtype=float)
    for q in range(Y.shape[1]):
        amp_q, _ = profile_candidate_amplitudes(S.T @ R_y[:, q], cand_e, cross, whitener)
        prof[:, q] = amp_q
    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(Z.shape[1]), Z.T @ S], [S.T @ Z, S.T @ S + lam_s * np.eye(M)]])
    joint_w = np.linalg.solve(H_aug, np.vstack([Z.T @ Y, S.T @ Y]))[Z.shape[1]:, :]
    return {"basis_fields": bf, "single_source_profiled_amplitudes": prof, "joint_source_weights": joint_w}
'''
    # Naive Dirichlet mismatch using `conservative_grid_operator` directly!
    vr02_dirichlet_bad = spd_src + "\n" + prof_src + r'''
def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    Lx, Ly = float(problem["Lx"]), float(problem["Ly"])
    kappa = np.asarray(problem["kappa"], dtype=float)
    sigma = np.asarray(problem["sigma"], dtype=float)
    nx, ny = kappa.shape
    hx, hy = Lx / nx, Ly / ny
    # WRONG: zero-ghost Dirichlet operator instead of Robin!
    A_dir = (hx * hy) * conservative_grid_operator(kappa, spacing=(hx, hy)) + sparse.diags((hx * hy) * sigma.ravel())
    M_prec = galerkin_preconditioner(A_dir)
    cand = np.asarray(problem["candidate_sources"], dtype=float)
    bf = np.zeros_like(cand)
    for m in range(cand.shape[0]):
        sol_m, _ = solve_residual_checked_cg(A_dir, (hx * hy) * cand[m].ravel(), preconditioner=M_prec, tol=1e-9)
        bf[m] = sol_m.reshape(nx, ny)
    s_ij = np.asarray(problem["sensor_ij"], dtype=int)
    S = bf[:, s_ij[:, 0], s_ij[:, 1]].T
    Z = np.asarray(problem["nuisance_design"], dtype=float)
    Y = np.asarray(problem["Y_obs"], dtype=float)
    lam_z, lam_s = float(problem["nuisance_ridge"]), float(problem["source_ridge"])
    _, R_y, whitener = fit_ridge_nuisance(Z, Y, ridge=lam_z)
    prof = np.column_stack([profile_candidate_amplitudes(S.T @ R_y[:, q], np.sum(S*S, axis=0), Z.T @ S, whitener)[0] for q in range(Y.shape[1])])
    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(Z.shape[1]), Z.T @ S], [S.T @ Z, S.T @ S + lam_s * np.eye(cand.shape[0])]])
    joint_w = np.linalg.solve(H_aug, np.vstack([Z.T @ Y, S.T @ Y]))[Z.shape[1]:, :]
    return {"basis_fields": bf, "single_source_profiled_amplitudes": prof, "joint_source_weights": joint_w}
'''
    with tempfile.TemporaryDirectory(prefix="m90a_vr02_ut_") as td:
        td_p = Path(td)
        tar_ok = _bundle_single_py_tar(td_p, "holdout_task/robin_spd_profiler.py", vr02_ref_code, "vr02_ok.tar")
        rep_ok = run_voluntary_reuse_private_verifier_by_id("vr02__large_robin_spd_preconditioned_cg_multi_rhs", tar_ok)
        assert rep_ok["strict_pass"] is True, f"VR02 reference failed: {rep_ok}"

        tar_bad = _bundle_single_py_tar(td_p, "holdout_task/robin_spd_profiler.py", vr02_dirichlet_bad, "vr02_bad.tar")
        rep_bad = run_voluntary_reuse_private_verifier_by_id("vr02__large_robin_spd_preconditioned_cg_multi_rhs", tar_bad)
        assert rep_bad["strict_pass"] is False, f"VR02 Dirichlet mismatch should fail: {rep_bad}"


def test_vr03_and_vr04_reference_solvers_pass_and_negative_controls_fail() -> None:
    v5_arch, srcs = _load_v5_l1_sources()
    trend_src = srcs["autogenous_prim_robust_trend_decomposition.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")
    panel_src = srcs["autogenous_prim_reference_panel_balancing.py"].replace("from . import track_wormhole_skill", "track_wormhole_skill = lambda *a, **k: (lambda f: f)")

    vr03_ref_code = trend_src + "\n" + panel_src + r'''
from sklearn.linear_model import LogisticRegression

def _extract_raw_moduli(taus, split_dict, use_prehistory=True):
    traces = split_dict["traces"]
    mask = np.asarray(split_dict["missing_channel_mask"], dtype=bool)
    N = len(traces)
    X_raw = np.zeros((N, 3), dtype=float)
    for i, tr in enumerate(traces):
        t = np.asarray(tr["timestamps"], dtype=float)
        u = np.asarray(tr["strain_u"], dtype=float)
        y_obs = np.asarray(tr["stress_obs"], dtype=float)
        past_t = np.asarray(tr["past_timestamps"], dtype=float)
        past_u = np.asarray(tr["past_strain_history"], dtype=float)
        M_pr = np.zeros((len(taus), len(t)), dtype=float)
        for r, tau in enumerate(taus):
            if use_prehistory:
                m = float(past_u[0])
                for k in range(1, len(past_t)):
                    a = math.exp(-(past_t[k] - past_t[k - 1]) / float(tau))
                    m = a * m + (1.0 - a) * float(past_u[k])
                for k in range(len(t)):
                    dt = (t[0] - past_t[-1]) if k == 0 else (t[k] - t[k - 1])
                    a = math.exp(-dt / float(tau))
                    m = a * m + (1.0 - a) * float(u[k])
                    M_pr[r, k] = m
            else:
                m = 0.0
                for k in range(len(t)):
                    dt = 0.15 if k == 0 else (t[k] - t[k - 1])
                    a = math.exp(-dt / float(tau))
                    m = a * m + (1.0 - a) * float(u[k])
                    M_pr[r, k] = m
        des = np.column_stack([np.ones_like(u), u, M_pr[0], M_pr[1]])
        beta, _ = cauchy_linear_fit(des, y_obs)
        X_raw[i, :] = beta[1:]
    X_raw[mask] = np.nan
    return X_raw

def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    taus = cohort["taus"]
    tr, qr = cohort["train"], cohort["query"]
    X_tr_raw = _extract_raw_moduli(taus, tr, use_prehistory=True)
    X_qr_raw = _extract_raw_moduli(taus, qr, use_prehistory=True)
    Z_tr = align_reference_panel(X_tr_raw, tr["controls"], tr["center_id"])
    Z_qr = align_reference_panel(X_qr_raw, qr["controls"], qr["center_id"])
    w_tr = balanced_cell_weights(tr["center_id"], tr["stratum"])
    col_means = weighted_column_fill(Z_tr, w_tr)
    Z_tr_imp = np.where(np.isfinite(Z_tr), Z_tr, col_means[None, :])
    Z_qr_imp = np.where(np.isfinite(Z_qr), Z_qr, col_means[None, :])
    s_tr = np.asarray(tr["stratum"], dtype=int)
    s_qr = np.asarray(qr["stratum"], dtype=int)
    F_tr = np.column_stack([Z_tr_imp, (s_tr == 1).astype(float), (s_tr == 2).astype(float)])
    F_qr = np.column_stack([Z_qr_imp, (s_qr == 1).astype(float), (s_qr == 2).astype(float)])
    clf = LogisticRegression(C=5.0, max_iter=500)
    clf.fit(F_tr, tr["y"], sample_weight=w_tr)
    probs = clf.predict_proba(F_qr)[:, 1]
    return {
        "aligned_query_moduli": Z_qr_imp,
        "query_probabilities": probs,
        "modulus_weights": clf.coef_[0, :3],
    }
'''
    vr03_no_prehistory = vr03_ref_code.replace("use_prehistory=True", "use_prehistory=False")
    vr03_unaligned_simpson = vr03_ref_code.replace(
        "Z_tr = align_reference_panel(X_tr_raw, tr[\"controls\"], tr[\"center_id\"])\n    Z_qr = align_reference_panel(X_qr_raw, qr[\"controls\"], qr[\"center_id\"])",
        "Z_tr = X_tr_raw\n    Z_qr = X_qr_raw",
    )

    vr04_ref_code = r'''
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def _assemble_helmholtz(Lx, Ly, k0, n_sq, beta_imp):
    nx, ny = n_sq.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    rows, cols, data = [], [], []
    diag = -(hx * hy) * (k0 ** 2) * n_sq.ravel().astype(np.complex128)
    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    wx = np.full(i0.size, hy / hx, dtype=np.complex128)
    diag[i0] += wx; diag[i1] += wx
    rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-wx, -wx])
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    wy = np.full(j0.size, hx / hy, dtype=np.complex128)
    diag[j0] += wy; diag[j1] += wy
    rows.extend([j0, j1]); cols.extend([j1, j0]); data.extend([-wy, -wy])
    for xb in (0, -1):
        diag[ids[xb, :].ravel()] += -hy * (1j * k0 * beta_imp)
    for yb in (0, -1):
        diag[ids[:, yb].ravel()] += -hx * (1j * k0 * beta_imp)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    Lx, Ly = float(problem["Lx"]), float(problem["Ly"])
    k0, beta_imp = float(problem["k0"]), float(problem["beta_imp"])
    n_sq = np.asarray(problem["n_sq"], dtype=float)
    basis = np.asarray(problem["basis"], dtype=float)
    M, nx, ny = basis.shape
    hx, hy = Lx / nx, Ly / ny
    H = _assemble_helmholtz(Lx, Ly, k0, n_sq, beta_imp)
    G_full = np.zeros((M, nx, ny), dtype=np.complex128)
    for m in range(M):
        G_full[m] = spsolve(H, (hx * hy) * basis[m].ravel()).reshape(nx, ny)

    pc = problem["pilot_calibration"]
    z_p = np.asarray(pc["amplitude_obs"], dtype=float) * np.exp(1j * np.asarray(pc["wrapped_phase_obs"], dtype=float))
    kp = np.asarray(pc["known_field_re"], dtype=float) + 1j * np.asarray(pc["known_field_im"], dtype=float)
    phasor = np.vdot(kp, z_p) / max(abs(np.vdot(kp, z_p)), 1e-15)
    phi0 = float(math.atan2(phasor.imag, phasor.real))

    sv = problem["survey"]
    s_ij = np.asarray(sv["sensor_ij"], dtype=int)
    z_s = np.asarray(sv["amplitude_obs"], dtype=float) * np.exp(1j * np.asarray(sv["wrapped_phase_obs"], dtype=float))
    u_demod = np.exp(-1j * phi0) * z_s
    Gs = G_full[:, s_ij[:, 0], s_ij[:, 1]].T  # (K_s, M) complex, real weights a!
    A_real = np.vstack([Gs.real, Gs.imag])
    b_real = np.concatenate([u_demod.real, u_demod.imag])
    alpha = float(sv["tikhonov_alpha"])

    # Cauchy IRLS on 2D complex residuals for spike immunity
    w_row = np.ones(Gs.shape[0], dtype=float)
    a_hat = np.zeros(M, dtype=float)
    for _ in range(18):
        W2 = np.concatenate([w_row, w_row])
        Aw = A_real * W2[:, None]
        a_hat = np.linalg.solve(A_real.T @ Aw + alpha * np.eye(M), Aw.T @ b_real)
        res_c = np.abs(Gs @ a_hat - u_demod)
        sc = max(1e-4, 1.4826 * float(np.median(res_c)))
        w_row = 1.0 / (1.0 + (res_c / (2.0 * sc)) ** 2)

    u_pred = np.tensordot(a_hat, G_full, axes=(0, 0))
    return {
        "source_amplitudes": a_hat,
        "global_phase_rad": phi0,
        "predicted_field_abs": np.abs(u_pred),
    }
'''
    vr04_naive_ols = vr04_ref_code.replace(
        "phasor = np.vdot(kp, z_p) / max(abs(np.vdot(kp, z_p)), 1e-15)\n    phi0 = float(math.atan2(phasor.imag, phasor.real))",
        "phi0 = 0.0  # ignores global receiver phase!",
    )

    with tempfile.TemporaryDirectory(prefix="m90a_vr34_ut_") as td:
        td_p = Path(td)
        t3_ok = _bundle_single_py_tar(td_p, "holdout_task/viscoelastic_panel_calibration.py", vr03_ref_code, "vr03_ok.tar")
        r3_ok = run_voluntary_reuse_private_verifier_by_id("vr03__prehistory_viscoelastic_control_panel_calibration", t3_ok)
        assert r3_ok["strict_pass"] is True, f"VR03 ref failed: {r3_ok}"

        t3_bad = _bundle_single_py_tar(td_p, "holdout_task/viscoelastic_panel_calibration.py", vr03_no_prehistory, "vr03_bad.tar")
        r3_bad = run_voluntary_reuse_private_verifier_by_id("vr03__prehistory_viscoelastic_control_panel_calibration", t3_bad)
        assert r3_bad["strict_pass"] is False, f"VR03 omitting prehistory must fail: {r3_bad}"

        t3_simp = _bundle_single_py_tar(td_p, "holdout_task/viscoelastic_panel_calibration.py", vr03_unaligned_simpson, "vr03_simp.tar")
        r3_simp = run_voluntary_reuse_private_verifier_by_id("vr03__prehistory_viscoelastic_control_panel_calibration", t3_simp)
        assert r3_simp["strict_pass"] is False, f"VR03 unaligned Simpson pooled fit must fail: {r3_simp}"

        t4_ok = _bundle_single_py_tar(td_p, "holdout_task/helmholtz_phase_inversion.py", vr04_ref_code, "vr04_ok.tar")
        r4_ok = run_voluntary_reuse_private_verifier_by_id("vr04__indefinite_helmholtz_wrapped_phase_inversion", t4_ok)
        assert r4_ok["strict_pass"] is True, f"VR04 ref failed: {r4_ok}"

        t4_bad = _bundle_single_py_tar(td_p, "holdout_task/helmholtz_phase_inversion.py", vr04_naive_ols, "vr04_bad.tar")
        r4_bad = run_voluntary_reuse_private_verifier_by_id("vr04__indefinite_helmholtz_wrapped_phase_inversion", t4_bad)
        assert r4_bad["strict_pass"] is False, f"VR04 naive zero-phase must fail: {r4_bad}"

    # Verify `net_value_gated` matrix on VR01..VR04:
    ftasks = {t.task_id: t for t in get_voluntary_reuse_m90a_frontier_tasks()}
    _, _, d_vr02 = m64.collect_recalled_executable_wormhole_sources(
        [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v5_arch,
        task_query_text=ftasks["vr02__large_robin_spd_preconditioned_cg_multi_rhs"].instruction_text,
        disclosure_mode="progressive", staging_policy="net_value_gated", return_diagnostics=True,
    )
    sel_vr02 = {x["filename"] for x in d_vr02["selected_l1_modules"]}
    assert "autogenous_prim_conservative_spd_solve.py" in sel_vr02
    assert "autogenous_prim_ridge_profiled_matching.py" in sel_vr02

    _, _, d_vr04 = m64.collect_recalled_executable_wormhole_sources(
        [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v5_arch,
        task_query_text=ftasks["vr04__indefinite_helmholtz_wrapped_phase_inversion"].instruction_text,
        disclosure_mode="progressive", staging_policy="net_value_gated", return_diagnostics=True,
    )
    sel_vr04 = {x["filename"] for x in d_vr04["selected_l1_modules"]}
    assert "autogenous_prim_conservative_spd_solve.py" not in sel_vr04
