"""Pre-Registration Hygiene & Reference-Solver Verification for `Hegel-OOD-Holdout-12` (`M7.3`).

Verifies BEFORE any LLM evaluation:
1. All 12 tasks (`ho01..ho06` Wave-1 + `ho07..ho12` Wave-2) are 100% disjoint from `TB-Science`, `BioMysteryBench`,
   and `pt01..pt06`, and `FrontierBenchmarkTask` wrappers physically exclude `private_verifier_py`.
2. Every `public_setup_py` AND every air-gapped `private_verifier_py` runs cleanly inside `hegel_clean_scipy_sandbox:v1`
   (NumPy 2.x / SciPy isolated `python3 -I` mode) and achieves `12/12` `strict_pass == True` under genuine mathematical
   reference solvers (while rejecting empty/broken submissions).
"""

from __future__ import annotations

from pathlib import Path
import tarfile
import tempfile

from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    build_holdout_ood_12_tasks,
    compute_holdout_suite_preregistration_sha256,
    holdout_tasks_as_frontier_waves,
    run_holdout_private_verifier,
)


REFERENCE_SOLVERS: dict[str, str] = {
    "ho01__crosswell_anisotropic_clock_tomography": r'''
import math, numpy as np
from scipy.optimize import least_squares

def invert_plume_and_clocks(survey: dict) -> dict:
    W = float(survey["W_m"]); H = float(survey["H_m"])
    zs = np.asarray(survey["z_src_m"], float)
    zr = np.asarray(survey["z_rcv_m"], float)
    T = np.asarray(survey["T_obs_s"], float)
    R = float(survey.get("R_plume_m", 5.5))
    ns, nr = len(zs), len(zr)
    dz = zr[None, :] - zs[:, None]
    L = np.hypot(W, dz)
    sin2 = (dz / L) ** 2

    # Step 1: Robust background + clock initialization via soft_l1
    def bg_res(p):
        s0, s1 = p[0], p[1]
        clk = p[2:] - np.mean(p[2:])
        pred = L * (s0 - s1 * sin2) + clk[None, :]
        return (pred - T).ravel()

    p0 = np.zeros(2 + nr)
    p0[0] = float(np.median(T / L))
    r_bg = least_squares(bg_res, p0, loss="cauchy", f_scale=1.5e-4)
    s0_i, s1_i = r_bg.x[0], r_bg.x[1]
    clk_i = r_bg.x[2:] - np.mean(r_bg.x[2:])
    res_mat = np.clip(T - (L * (s0_i - s1_i * sin2) + clk_i[None, :]), -0.002, 0.002)

    u_q = np.linspace(0.0, 1.0, 41)
    xq = u_q[None, None, :] * W
    zq = zs[:, None, None] + u_q[None, None, :] * dz[:, :, None]

    def ray_gauss(xp, zp):
        g = np.exp(-0.5 * ((xq - xp) ** 2 + (zq - zp) ** 2) / (R ** 2))
        return 0.5 * np.sum((g[:, :, :-1] + g[:, :, 1:]) * np.diff(u_q)[None, None, :], axis=2) * L

    best_score, best_xz = -1e18, (0.5 * W, 0.5 * H)
    for xp in np.linspace(6.0, W - 6.0, 13):
        for zp in np.linspace(10.0, H - 10.0, 17):
            B = ray_gauss(xp, zp)
            Bc = B - np.mean(B, axis=0, keepdims=True)
            rc = res_mat - np.mean(res_mat, axis=0, keepdims=True)
            sc = float(np.sum(Bc * rc) / (np.linalg.norm(Bc) + 1e-12))
            if sc > best_score:
                best_score, best_xz = sc, (float(xp), float(zp))

    u_fine = np.linspace(0.0, 1.0, 61)
    xqf = u_fine[None, None, :] * W
    zqf = zs[:, None, None] + u_fine[None, None, :] * dz[:, :, None]

    def full_res(p):
        xp, zp, s0, ep, ds = p[:5]
        clk = p[5:] - np.mean(p[5:])
        g = np.exp(-0.5 * ((xqf - xp) ** 2 + (zqf - zp) ** 2) / (R ** 2))
        tpl = ds * 0.5 * np.sum((g[:, :, :-1] + g[:, :, 1:]) * np.diff(u_fine)[None, None, :], axis=2) * L
        pred = L * s0 * (1.0 - ep * sin2) + tpl + clk[None, :]
        return (pred - T).ravel()

    q0 = np.r_[best_xz[0], best_xz[1], s0_i, max(0.01, s1_i / max(s0_i, 1e-6)), 6.5e-5, clk_i]
    lb = np.r_[2.0, 2.0, 1e-5, 0.0, 0.0, np.full(nr, -0.01)]
    ub = np.r_[W - 2.0, H - 2.0, 5e-3, 0.5, 1e-3, np.full(nr, 0.01)]
    sol = least_squares(full_res, q0, bounds=(lb, ub), loss="cauchy", f_scale=8e-5, max_nfev=300)
    clk_f = sol.x[5:] - np.mean(sol.x[5:])
    return {"plume_xz_m": [float(sol.x[0]), float(sol.x[1])], "rcv_clock_bias_s": clk_f.tolist()}
''',

    "ho02__inhibited_enzyme_rational_kinetics": r'''
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

def _sim(params, y0, t_eval, fa, w0, w1):
    Vm1, Km1, ap, Vm2, Km2, kl = params
    def rhs(t, y):
        S, I, P = np.maximum(y, 0.0)
        uf = fa if (w0 <= t <= w1) else 0.0
        v1 = Vm1 * S / (Km1 + S + ap * P)
        v2 = Vm2 * I / (Km2 + I)
        return [-v1 + uf, v1 - v2 - kl * I, v2]
    # Split integration at discontinuities w0, w1 for accuracy & speed
    pts = sorted({float(t_eval[0]), float(t_eval[-1]), float(np.clip(w0, t_eval[0], t_eval[-1])), float(np.clip(w1, t_eval[0], t_eval[-1]))})
    out = np.zeros((len(t_eval), 3))
    cur_y = np.maximum(np.asarray(y0, float), 0.0)
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a:
            continue
        mask = (t_eval >= a - 1e-12) & (t_eval <= b + 1e-12)
        te = t_eval[mask]
        sol = solve_ivp(rhs, [a, b], cur_y, t_eval=te, rtol=1e-6, atol=1e-8)
        if len(te):
            out[mask] = sol.y.T
        sol_end = solve_ivp(rhs, [a, b], cur_y, rtol=1e-6, atol=1e-8)
        cur_y = sol_end.y[:, -1]
    return out

def identify_and_forecast(dataset: dict) -> dict:
    trajs = dataset["train_trajectories"]
    n_tr = len(trajs)
    p_init = np.r_[1.8, 0.6, 1.2, 1.4, 0.5, 0.14, np.ravel([tr["y0_approx"] for tr in trajs])]
    lb = np.r_[0.1, 0.05, 0.01, 0.1, 0.05, 0.005, np.zeros(3 * n_tr)]
    ub = np.r_[8.0, 5.0, 8.0, 8.0, 5.0, 2.0, np.full(3 * n_tr, 10.0)]

    def res(theta):
        par = theta[:6]
        errs = []
        for k, tr in enumerate(trajs):
            y0k = theta[6 + 3 * k: 6 + 3 * (k + 1)]
            w0, w1 = tr["feed_window"]
            pred = _sim(par, y0k, np.asarray(tr["t_obs"], float), float(tr["feed_amp"]), float(w0), float(w1))
            errs.append((pred - np.asarray(tr["y_obs"], float)).ravel())
        return np.concatenate(errs)

    opt = least_squares(res, p_init, bounds=(lb, ub), max_nfev=120)
    par = opt.x[:6]
    names = ("Vmax1", "Km1", "alpha_p", "Vmax2", "Km2", "k_leak")
    w0, w1 = dataset["test_feed_window"]
    fc = _sim(par, dataset["y0_test"], np.asarray(dataset["t_test"], float), float(dataset["test_feed_amp"]), float(w0), float(w1))
    return {"params": {k: float(v) for k, v in zip(names, par)}, "forecast_SIP": fc}
''',

    "ho03__high_peclet_upwind_plume_inversion": r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized
from scipy.optimize import least_squares

def _bernoulli(z):
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    sm = np.abs(z) < 1e-4
    out[sm] = 1.0 - 0.5 * z[sm] + (z[sm] ** 2) / 12.0
    zs = z[~sm]
    out[~sm] = np.where(zs > 40.0, zs * np.exp(-zs), np.where(zs < -40.0, -zs, zs / np.expm1(zs)))
    return out

def _build_sg_matrix(ux_face, uy_face, D, kappa, h):
    ux = np.asarray(ux_face, float); uy = np.asarray(uy_face, float)
    N = ux.shape[1]
    inv_h2 = D / (h * h)
    pr = ux[1:, :] * h / D; pl = ux[:-1, :] * h / D
    pt = uy[:, 1:] * h / D; pb = uy[:, :-1] * h / D
    diag = kappa + inv_h2 * (_bernoulli(-pr) + _bernoulli(pl) + _bernoulli(-pt) + _bernoulli(pb))
    idx = np.arange(N * N).reshape(N, N)
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    # Right
    rows.append(idx[:-1, :].ravel()); cols.append(idx[1:, :].ravel()); vals.append((-inv_h2 * _bernoulli(pr[:-1, :])).ravel())
    # Left
    rows.append(idx[1:, :].ravel()); cols.append(idx[:-1, :].ravel()); vals.append((-inv_h2 * _bernoulli(-pl[1:, :])).ravel())
    # Top
    rows.append(idx[:, :-1].ravel()); cols.append(idx[:, 1:].ravel()); vals.append((-inv_h2 * _bernoulli(pt[:, :-1])).ravel())
    # Bottom
    rows.append(idx[:, 1:].ravel()); cols.append(idx[:, :-1].ravel()); vals.append((-inv_h2 * _bernoulli(-pb[:, 1:])).ravel())
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N))

def forward_peclet_field(ux_face, uy_face, diffusivity, kappa, source_map, h):
    N = np.asarray(source_map).shape[0]
    A = _build_sg_matrix(ux_face, uy_face, float(diffusivity), float(kappa), float(h))
    solve = factorized(A.tocsc())
    return solve(np.asarray(source_map, float).ravel()).reshape(N, N)

def invert_point_source(problem: dict) -> dict:
    N = int(problem["N"]); L = float(problem["L"]); h = float(problem["h"])
    A = _build_sg_matrix(problem["ux_face"], problem["uy_face"], float(problem["D"]), float(problem["kappa"]), h)
    # Adjoint Green's functions at the 18 sensors: A^T g_m = e_{s_m}! Only 18 sparse solves total!
    solve_T = factorized(A.T.tocsc())
    s_ij = np.asarray(problem["sensor_ij"], int)
    c_obs = np.asarray(problem["c_obs"], float)
    M = len(s_ij)
    G = np.zeros((M, N, N), dtype=float)
    for m in range(M):
        rhs = np.zeros(N * N)
        rhs[s_ij[m, 0] * N + s_ij[m, 1]] = 1.0
        G[m] = solve_T(rhs).reshape(N, N)

    xc = (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    sig = float(problem["sigma_s"])

    def unit_resp(xs, ys):
        kern = np.exp(-0.5 * ((X - xs) ** 2 + (Y - ys) ** 2) / (sig ** 2))
        return np.tensordot(G, kern, axes=([1, 2], [0, 1]))

    best_loss, best_p = 1e18, (0.5 * L, 0.5 * L, 1.0)
    for xs in np.linspace(0.12 * L, 0.88 * L, 15):
        for ys in np.linspace(0.12 * L, 0.88 * L, 15):
            u = unit_resp(xs, ys)
            ratios = c_obs[u > 1e-4] / u[u > 1e-4]
            q0 = float(np.median(ratios)) if len(ratios) else 1.0
            r = np.abs(q0 * u - c_obs)
            loss = float(np.sum(np.sort(r)[: M - 3] ** 2))
            if loss < best_loss:
                best_loss, best_p = loss, (float(xs), float(ys), max(0.1, q0))

    opt = least_squares(
        lambda p: p[2] * unit_resp(p[0], p[1]) - c_obs,
        np.array(best_p),
        bounds=([0.02 * L, 0.02 * L, 0.01], [0.98 * L, 0.98 * L, 50.0]),
        loss="cauchy",
        f_scale=0.015,
    )
    return {"source_xy": [float(opt.x[0]), float(opt.x[1])], "strength_Q": float(opt.x[2])}
''',

    "ho04__magnetometer_hard_soft_iron_calibration": r'''
import math, numpy as np
from scipy.optimize import least_squares

def _tilt_mats(rp):
    rolls = rp[:, 0]; pitches = rp[:, 1]
    cr, sr = np.cos(rolls), np.sin(rolls)
    cp, sp = np.cos(pitches), np.sin(pitches)
    K = len(rp)
    M = np.zeros((K, 3, 3))
    # Ry(pitch) @ Rx(roll)
    M[:, 0, 0] = cp; M[:, 0, 1] = sp * sr; M[:, 0, 2] = sp * cr
    M[:, 1, 1] = cr; M[:, 1, 2] = -sr
    M[:, 2, 0] = -sp; M[:, 2, 1] = cp * sr; M[:, 2, 2] = cp * cr
    return M

def _sym_det1(v5):
    # Symmetric 3x3 from 5 params via matrix exp of trace-zero symmetric!
    a, b, d, e, f = v5
    H = np.array([[a, d, e], [d, b, f], [e, f, -(a + b)]], float)
    w, V = np.linalg.eigh(H)
    return (V * np.exp(w)) @ V.T

def calibrate_and_estimate_heading(session: dict) -> dict:
    m_cal = np.asarray(session["calib_m_raw"], float)
    rp_cal = np.asarray(session["calib_roll_pitch_rad"], float)
    Rt_cal = _tilt_mats(rp_cal)

    b0 = np.median(m_cal, axis=0)
    r0 = float(np.median(np.linalg.norm(m_cal - b0, axis=1)))
    x0 = np.r_[np.zeros(5), b0, 0.5 * r0, 0.85 * r0]

    def res(x):
        S_inv = _sym_det1(x[:5])
        bh = x[5:8]
        Bn, Bd = x[8], x[9]
        c = (m_cal - bh) @ S_inv.T
        lev = np.einsum("kij,kj->ki", Rt_cal, c)
        horiz = np.hypot(lev[:, 0], lev[:, 1])
        return np.r_[horiz - Bn, lev[:, 2] - Bd]

    lb = np.r_[np.full(5, -1.0), np.full(3, -200.0), 1.0, -100.0]
    ub = np.r_[np.full(5, 1.0), np.full(3, 200.0), 100.0, 100.0]
    opt = least_squares(res, x0, bounds=(lb, ub), loss="cauchy", f_scale=0.25, max_nfev=300)
    S_inv = _sym_det1(opt.x[:5])
    bh = opt.x[5:8]

    m_te = np.asarray(session["test_m_raw"], float)
    rp_te = np.asarray(session["test_roll_pitch_rad"], float)
    Rt_te = _tilt_mats(rp_te)
    lev_te = np.einsum("kij,kj->ki", Rt_te, (m_te - bh) @ S_inv.T)
    hdg = np.degrees(np.arctan2(-lev_te[:, 1], lev_te[:, 0]))
    return {"b_hard": bh.tolist(), "S_soft": np.linalg.inv(S_inv).tolist(), "heading_deg": hdg.tolist()}
''',

    "ho05__latent_confounded_graphical_precision": r'''
import numpy as np

def recover_sparse_precision_support(X_samples: np.ndarray) -> dict:
    X = np.asarray(X_samples, float)
    n, p = X.shape
    Xc = X - np.mean(X, axis=0, keepdims=True)
    Sig_hat = (Xc.T @ Xc) / n
    off = ~np.eye(p, dtype=bool)
    w0, V0 = np.linalg.eigh(Sig_hat)
    F = V0[:, -2:] * np.sqrt(np.maximum(w0[-2:] - np.median(w0[:-2]), 0.01))
    S = np.diag(1.0 / np.maximum(np.diag(Sig_hat - F @ F.T), 0.1))
    for _ in range(35):
        S_inv = np.linalg.inv(S)
        W_inv = np.linalg.inv(S_inv + F @ F.T)
        Beta = F.T @ W_inv
        C_h = np.eye(2) - Beta @ F
        EH = Xc @ Beta.T
        EHH = (EH.T @ EH) / n + C_h
        F = (Sig_hat @ Beta.T) @ np.linalg.inv(EHH)
        Sig_Z = Sig_hat - F @ (Beta @ Sig_hat) - (Sig_hat @ Beta.T) @ F.T + F @ EHH @ F.T
        Sig_Z = 0.5 * (Sig_Z + Sig_Z.T) + 0.005 * np.eye(p)
        S_raw = np.linalg.inv(Sig_Z)
        d = np.sqrt(np.maximum(np.diag(S_raw), 1e-8))
        pcorr = np.abs(S_raw) / np.outer(d, d)
        S = S_raw.copy()
        S[off & (pcorr < 0.10)] = 0.0
        ew = np.linalg.eigvalsh(S)
        if ew[0] <= 0.05:
            S += (0.1 - ew[0]) * np.eye(p)
    d = np.sqrt(np.maximum(np.diag(S_raw), 1e-8))
    pcorr = np.abs(S_raw) / np.outer(d, d)
    adj = off & (pcorr >= 0.125)
    return {"adjacency": adj, "sparse_precision": S}
''',

    "ho06__plate_evaporation_deconfounded_ic50": r'''
import numpy as np
from scipy.optimize import least_squares

def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    by_cmp: dict[str, list[tuple[float, float]]] = {}
    for pl in plates:
        wells = pl["wells"]
        pos_y = [float(w["fluorescence"]) for w in wells if w["well_type"] == "POS_KILL"]
        B = float(np.median(pos_y))
        dmso = [w for w in wells if w["well_type"] == "DMSO"]
        dr = np.array([float(w["row"]) for w in dmso])
        dc = np.array([float(w["col"]) for w in dmso])
        dy = np.array([float(w["fluorescence"]) - B for w in dmso])
        de = np.minimum(np.minimum(dr, 15.0 - dr), np.minimum(dc, 23.0 - dc))

        def g_res(p):
            smax, a, tau, br, bc = p
            g = 1.0 + a * np.exp(-de / tau) + br * (dr / 15.0) + bc * (dc / 23.0)
            return smax * g - dy

        opt_g = least_squares(
            g_res,
            [float(np.median(dy)), 0.35, 1.8, 0.0, 0.0],
            bounds=([100.0, -0.5, 0.4, -0.5, -0.5], [10000.0, 1.5, 8.0, 0.5, 0.5]),
            loss="cauchy",
            f_scale=20.0,
        )
        smax, a, tau, br, bc = opt_g.x
        for w in wells:
            if w["well_type"] == "COMPOUND":
                r, c = float(w["row"]), float(w["col"])
                d_e = min(r, 15.0 - r, c, 23.0 - c)
                g_rc = 1.0 + a * np.exp(-d_e / tau) + br * (r / 15.0) + bc * (c / 23.0)
                viab = (float(w["fluorescence"]) - B) / (smax * g_rc)
                by_cmp.setdefault(str(w["compound_id"]), []).append((float(w["log10_conc_uM"]), float(viab)))

    out: dict[str, dict[str, float]] = {}
    for cid, pts in sorted(by_cmp.items()):
        arr = np.array(pts, float)
        lc, vb = arr[:, 0], arr[:, 1]
        opt_h = least_squares(
            lambda p: 1.0 / (1.0 + 10.0 ** (p[1] * (lc - p[0]))) - vb,
            [0.0, 1.0],
            bounds=([-4.0, 0.2], [4.0, 4.0]),
            loss="cauchy",
            f_scale=0.03,
        )
        out[cid] = {"log10_ic50_uM": float(opt_h.x[0]), "hill_slope": float(opt_h.x[1])}
    return out
''',

    "ho07__thermoacoustic_speed_shifted_ring": r'''
import math, numpy as np
from scipy.optimize import least_squares

def reconstruct_two_absorbers_mm(scan: dict) -> dict:
    fs = float(scan["fs_hz"])
    s_xy = np.asarray(scan["sensor_xy_mm"], float)
    pin_xy = np.asarray(scan["calib_pin_xy_mm"], float)
    sig = np.asarray(scan["signals"], float)
    n_t, n_s = sig.shape
    t = np.arange(n_t) / fs
    tau = 2.2e-7
    d_pin = np.linalg.norm(s_xy - pin_xy[None, :], axis=1) * 1e-3

    def pin_coh(c):
        tot = 0.0
        for j in range(n_s):
            dt = t - d_pin[j] / c
            w = -(dt / tau) * np.exp(-0.5 * (dt / tau) ** 2) / math.sqrt(max(d_pin[j], 0.005))
            tot += abs(float(np.dot(sig[:, j], w))) / (float(np.linalg.norm(w)) + 1e-12)
        return tot

    cs = np.linspace(1430.0, 1550.0, 241)
    c_init = float(cs[int(np.argmax([pin_coh(c) for c in cs]))])

    # Estimate signed channel gains from pin response (amplitude 0.55)
    gains = np.ones(n_s)
    sig_clean = sig.copy()
    for j in range(n_s):
        dt = t - d_pin[j] / c_init
        w = 0.55 * (-(dt / tau) * np.exp(-0.5 * (dt / tau) ** 2)) / math.sqrt(max(d_pin[j], 0.005))
        gj = float(np.dot(sig[:, j], w) / np.dot(w, w))
        gains[j] = gj
        sig_clean[:, j] = (sig[:, j] - gj * w) / gj

    # Matched-filter N-wave map on coarse 1.0 mm grid
    k_half = 18
    kt = (np.arange(-k_half, k_half + 1) / fs)
    kw = -(kt / tau) * np.exp(-0.5 * (kt / tau) ** 2)
    mf = np.array([np.convolve(sig_clean[:, j], kw[::-1], mode="same") for j in range(n_s)])  # (16, n_t)

    gx = np.linspace(-30.0, 30.0, 121)
    GX, GY = np.meshgrid(gx, gx, indexing="ij")
    mask = (GX ** 2 + GY ** 2) <= 31.0 ** 2
    img = np.zeros_like(GX)
    for j in range(n_s):
        dm = np.hypot(GX - s_xy[j, 0], GY - s_xy[j, 1]) * 1e-3
        idx = np.clip(np.rint(dm / c_init * fs).astype(int), 0, n_t - 1)
        img += mf[j, idx] * np.sqrt(np.maximum(dm, 0.005))
    img[~mask] = -1e9
    i1, j1 = np.unravel_index(int(np.argmax(img)), img.shape)
    p1_0 = np.array([gx[i1], gx[j1]])
    img2 = img.copy()
    img2[(GX - p1_0[0]) ** 2 + (GY - p1_0[1]) ** 2 < 36.0] = -1e9
    i2, j2 = np.unravel_index(int(np.argmax(img2)), img2.shape)
    p2_0 = np.array([gx[i2], gx[j2]])

    # Joint waveform nonlinear least squares over (c, gains_16, x1, y1, a1, x2, y2, a2)
    def full_res(theta):
        c = theta[0]
        gs = theta[1:17]
        x1, y1, a1, x2, y2, a2 = theta[17:]
        srcs = ((pin_xy[0], pin_xy[1], 0.55), (x1, y1, a1), (x2, y2, a2))
        pred = np.zeros_like(sig)
        for sx, sy, amp in srcs:
            dm = np.hypot(s_xy[:, 0] - sx, s_xy[:, 1] - sy) * 1e-3
            dt = t[:, None] - dm[None, :] / c
            pred += amp * (-(dt / tau) * np.exp(-0.5 * (dt / tau) ** 2)) / np.sqrt(np.maximum(dm[None, :], 0.005))
        return (pred * gs[None, :] - sig).ravel()

    th0 = np.r_[c_init, gains, p1_0[0], p1_0[1], 0.9, p2_0[0], p2_0[1], 0.8]
    opt = least_squares(full_res, th0, max_nfev=80)
    c_f = float(opt.x[0])
    pts = sorted([[float(opt.x[17]), float(opt.x[18])], [float(opt.x[20]), float(opt.x[21])]], key=lambda q: q[0])
    return {"sound_speed_m_per_s": c_f, "absorbers_xy_mm": pts}
''',

    "ho08__prony_viscoelastic_memory_inversion": r'''
import numpy as np
from scipy.optimize import least_squares

def _prony_basis(t, eps, taus):
    dt = np.diff(t); deps = np.diff(eps)
    M = len(taus)
    B = np.zeros((len(t), M), float)
    for k, tk in enumerate(taus):
        z = np.zeros(len(t))
        for i in range(len(dt)):
            dec = np.exp(-dt[i] / tk)
            z[i + 1] = dec * z[i] + tk * (1.0 - dec) * (deps[i] / max(dt[i], 1e-12))
        B[:, k] = z
    return B

def fit_prony_and_predict_stress(experiment: dict) -> dict:
    t_tr = np.asarray(experiment["train_t"], float)
    e_tr = np.asarray(experiment["train_strain"], float)
    y_tr = np.asarray(experiment["train_stress_raw"], float)
    t_te = np.asarray(experiment["test_t"], float)
    e_te = np.asarray(experiment["test_strain"], float)

    # Note: maximum relaxation tau <= 10s so G_inf is cleanly decoupled from linear thermal drift b1*t on [0, 30s]!
    taus = np.geomspace(0.05, 11.0, 26)
    B_tr = _prony_basis(t_tr, e_tr, taus)
    A = np.column_stack([e_tr, B_tr, np.ones_like(t_tr), t_tr])
    m = len(taus)
    lb = np.r_[0.0, np.zeros(m), -50.0, -5.0]
    ub = np.r_[200.0, np.full(m, 200.0), 50.0, 5.0]
    x0 = np.r_[10.0, np.full(m, 1.0), 0.0, 0.0]

    opt = least_squares(
        lambda x: np.r_[A @ x - y_tr, 0.008 * x[1 : 1 + m]],
        x0,
        bounds=(lb, ub),
        loss="cauchy",
        f_scale=0.04,
        max_nfev=250,
    )
    G_inf = float(opt.x[0])
    Gs = opt.x[1 : 1 + m]
    B_te = _prony_basis(t_te, e_te, taus)
    pred = G_inf * e_te + B_te @ Gs
    return {"G_inf": G_inf, "test_stress": pred.tolist()}
''',

    "ho09__shallow_water_sill_and_gate_inversion": r'''
import numpy as np
from scipy.optimize import minimize_scalar

def _rollout(sluice_high: bool, sill_h: float, T_steps: int, N: int, dx: float, dt: float):
    g = 9.81
    xc = (np.arange(N) + 0.5) * dx
    b = sill_h * np.exp(-0.5 * ((xc - 400.0) / 45.0) ** 2)
    eta0 = np.full(N, 2.0)
    eta0[:8] = 2.65 if sluice_high else 1.45
    h = eta0 - b; q = np.zeros(N)
    etas = np.zeros((T_steps, N)); etas[0] = h + b
    for tk in range(T_steps - 1):
        h_ext = np.r_[h[0], h, h[-1]]; b_ext = np.r_[b[0], b, b[-1]]; q_ext = np.r_[-q[0], q, -q[-1]]
        hL, hR = h_ext[:-1], h_ext[1:]; bL, bR = b_ext[:-1], b_ext[1:]; qL, qR = q_ext[:-1], q_ext[1:]
        uL = qL / np.maximum(hL, 1e-9); uR = qR / np.maximum(hR, 1e-9); bs = np.maximum(bL, bR)
        hLs = np.maximum(0.0, hL + bL - bs); hRs = np.maximum(0.0, hR + bR - bs)
        FL0, FL1 = hLs * uL, hLs * uL ** 2 + 0.5 * g * hLs ** 2
        FR0, FR1 = hRs * uR, hRs * uR ** 2 + 0.5 * g * hRs ** 2
        sm = np.maximum(np.maximum(np.abs(uL) + np.sqrt(g * hLs), np.abs(uR) + np.sqrt(g * hRs)), 1e-6)
        Fn0 = 0.5 * (FL0 + FR0) - 0.5 * sm * (hRs - hLs)
        Fn1 = 0.5 * (FL1 + FR1) - 0.5 * sm * (hRs * uR - hLs * uL)
        src1 = 0.5 * g * (hLs[1:] ** 2 - hRs[:-1] ** 2)
        h = h - (dt / dx) * (Fn0[1:] - Fn0[:-1])
        q = q - (dt / dx) * (Fn1[1:] - Fn1[:-1]) + (dt / dx) * src1
        etas[tk + 1] = h + b
    return etas

def invert_canal_regime(obs_bundle: dict) -> dict:
    T_steps = int(obs_bundle["T_steps"]); dt = float(obs_bundle["dt"]); dx = float(obs_bundle["dx"]); N = int(obs_bundle["N"])
    gauges = list(obs_bundle["gauge_cells"])
    t_idx = np.asarray(obs_bundle["obs_step_indices"], int)
    obs = np.asarray(obs_bundle["eta_obs"], float)
    best = None
    for mode in (False, True):
        res = minimize_scalar(
            lambda sh: float(np.mean((_rollout(mode, sh, T_steps, N, dx, dt)[np.ix_(t_idx, gauges)] - obs) ** 2)),
            bounds=(0.0, 0.75),
            method="bounded",
        )
        if best is None or res.fun < best[0]:
            best = (float(res.fun), mode, float(res.x))
    _, best_mode, best_sh = best
    full = _rollout(best_mode, best_sh, T_steps, N, dx, dt)
    return {"sluice_high_mode": bool(best_mode), "sill_height_m": best_sh, "eta_full": full}
''',

    "ho10__haldane_bioreactor_runaway_control": r'''
import numpy as np

def compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]:
    mem = dict(state_memory)
    V = float(obs["V_L"]); dt = float(obs["dt_h"]); t = float(obs["t_h"])
    S_m = float(obs["S_meas_g_L"]); T_m = float(obs["T_meas_C"]); X_m = float(obs["X_meas_g_L"])
    S_f = 0.55 * S_m + 0.45 * float(mem.get("S_f", S_m))
    T_f = 0.65 * T_m + 0.35 * float(mem.get("T_f", T_m))
    X_f = 0.55 * X_m + 0.45 * float(mem.get("X_f", X_m))
    dT_dt = (T_f - float(mem.get("T_f", T_f))) / max(dt, 1e-6)
    mem.update({"S_f": S_f, "T_f": T_f, "X_f": X_f})

    # Keep substrate strictly near optimal non-inhibited Haldane setpoint S_ref = 2.6 g/L (< S* ~ 3.4 g/L!)
    S_ref = 2.6
    # Approximate consumption rate (mu ~ 0.25) -> F_ff = (mu / Yxs) * X * V / (Sf - S_ref)
    mu_est = 0.26 * np.exp(0.065 * (T_f - 30.0)) * S_f / (0.85 + S_f + (S_f ** 2) / 14.0)
    F_cmd = (mu_est / 0.50) * X_f * V / max(45.0 - S_f, 10.0) + 0.35 * (S_ref - S_f)
    # Hard volume & thermal runaway safety barrier (accounting for 1-step delay!)
    rem_V = 11.88 - V
    F_cmd = min(F_cmd, max(0.0, rem_V / max(2.0 * dt, 1e-6)))
    T_pred = T_f + 0.12 * max(dT_dt, 0.0)
    if T_pred > 32.8:
        F_cmd *= max(0.0, (33.5 - T_pred) / 0.7)
    F_cmd = float(np.clip(F_cmd, 0.0, 2.5))

    # Regulate T around 32.2 C for high growth while enforcing T <= 34.5 C with full cooling above 32.6 C!
    if T_pred >= 32.6:
        uc = 1.0
    else:
        uc = float(np.clip(0.35 + 0.65 * (T_pred - 31.2) / 1.4, 0.0, 1.0))
    return {"feed_rate_L_per_h": F_cmd, "coolant_flow_frac": uc}, mem
''',

    "ho11__certified_capacitated_facility_bnb": r'''
import numpy as np
from scipy.optimize import linprog

def solve_and_certify_facilities(instance: dict) -> dict:
    m = int(instance["m"]); n = int(instance["n"])
    fcost = np.asarray(instance["fixed_cost"], float)
    cap = np.asarray(instance["capacity"], float)
    dem = np.asarray(instance["demand"], float)
    cost = np.asarray(instance["cost_matrix"], float)
    tot_dem = float(np.sum(dem))

    # Joint LP relaxation variables: z = [X_flat (m*n), y (m)]
    nv = m * n + m
    c_lp = np.r_[cost.ravel(), fcost]
    A_eq = np.zeros((n, nv))
    for j in range(n):
        A_eq[j, j : m * n : n] = 1.0
    # sum_j X[i, j] - cap[i] * y[i] <= 0
    A_ub = np.zeros((m, nv))
    for i in range(m):
        A_ub[i, i * n : (i + 1) * n] = 1.0
        A_ub[i, m * n + i] = -cap[i]
    b_ub = np.zeros(m)

    def solve_node_lp(fc_set, fo_set):
        if sum(cap[i] for i in range(m) if i not in fc_set) < tot_dem - 1e-9:
            return np.inf, None, None
        bounds = [(0.0, None)] * (m * n)
        for i in range(m):
            if i in fc_set:
                bounds.append((0.0, 0.0))
            elif i in fo_set:
                bounds.append((1.0, 1.0))
            else:
                bounds.append((0.0, 1.0))
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=dem, bounds=bounds, method="highs")
        if not res.success:
            return np.inf, None, None
        return float(res.fun), res.x[: m * n].reshape(m, n), res.x[m * n :]

    best_ub = np.inf
    best_y = None
    best_X = None

    # Pass 1: Find exact global MIP optimum via depth-first B&B
    def find_opt(fc_set, fo_set):
        nonlocal best_ub, best_y, best_X
        lb, X_lp, y_lp = solve_node_lp(fc_set, fo_set)
        if lb >= best_ub - 1e-7 or y_lp is None:
            return
        frac = [i for i in range(m) if (i not in fc_set) and (i not in fo_set) and min(y_lp[i], 1.0 - y_lp[i]) > 1e-6]
        if not frac:
            yi = np.rint(y_lp).astype(int)
            # Re-solve exact transport LP for integer yi
            lb_i, Xi, _ = solve_node_lp({i for i in range(m) if yi[i] == 0}, {i for i in range(m) if yi[i] == 1})
            if lb_i < best_ub:
                best_ub, best_y, best_X = lb_i, yi, Xi
            return
        # Round heuristic
        yh = (y_lp >= 0.35).astype(int)
        lb_h, Xh, _ = solve_node_lp({i for i in range(m) if yh[i] == 0}, {i for i in range(m) if yh[i] == 1})
        if lb_h < best_ub:
            best_ub, best_y, best_X = lb_h, yh, Xh
        if lb >= best_ub - 1e-7:
            return
        bi = max(frac, key=lambda i: min(y_lp[i], 1.0 - y_lp[i]))
        if y_lp[bi] >= 0.5:
            find_opt(fc_set, fo_set | {bi})
            find_opt(fc_set | {bi}, fo_set)
        else:
            find_opt(fc_set | {bi}, fo_set)
            find_opt(fc_set, fo_set | {bi})

    find_opt(set(), set())

    # Pass 2: Build minimal compacted proof partition tree for fixed `best_ub`!
    def certify(fc_set, fo_set):
        lb, _, y_lp = solve_node_lp(fc_set, fo_set)
        if lb >= best_ub - 1e-6:
            return [{"fixed_closed": sorted(fc_set), "fixed_open": sorted(fo_set), "lower_bound": float(min(lb, 1e9))}]
        free = [i for i in range(m) if (i not in fc_set) and (i not in fo_set)]
        bi = max(free, key=lambda i: min(y_lp[i], 1.0 - y_lp[i]))
        left = certify(fc_set | {bi}, fo_set)
        right = certify(fc_set, fo_set | {bi})
        if len(left) == 1 and len(right) == 1:
            merged_lb = min(left[0]["lower_bound"], right[0]["lower_bound"])
            if merged_lb >= best_ub - 1e-6:
                return [{"fixed_closed": sorted(fc_set), "fixed_open": sorted(fo_set), "lower_bound": float(merged_lb)}]
        return left + right

    leaves = certify(set(), set())
    return {
        "open_y": best_y.tolist(),
        "flows_X": best_X.tolist(),
        "upper_bound": float(best_ub),
        "proof_leaves": leaves,
    }
''',

    "ho12__multicopy_plasmid_cohort_consensus": r'''
import glob, json, os, numpy as np

CODON = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E","TGT":"C","TGC":"C","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G","TAA":"*","TAG":"*","TGA":"*",
}

def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

def audit_outbreak_cohort(cohort_dir: str) -> dict:
    with open(os.path.join(cohort_dir, "reference_panel.json")) as fh:
        wt_cds = json.load(fh)["reference_wildtype_cds"]
    iso_files = sorted(glob.glob(os.path.join(cohort_dir, "isolate_*.json")))
    by_tag: dict[str, list[tuple[float, str]]] = {}
    for fp in iso_files:
        data = json.load(open(fp))
        contigs = data["contigs"]
        chr_d = float(np.median([float(c["mean_depth"]) for c in contigs if not c["circular"]]))
        seen_in_iso = set()
        for c in contigs:
            if c["circular"] and len(c["sequence"]) >= 2400:
                cn = float(c["mean_depth"]) / chr_d
                tag = str(c["family_tag"])
                if tag not in seen_in_iso:
                    seen_in_iso.add(tag)
                    by_tag.setdefault(tag, []).append((cn, str(c["sequence"])))

    best_tag = max(by_tag.keys(), key=lambda k: len(by_tag[k]))
    entries = by_tag[best_tag]
    med_cn = float(np.median([e[0] for e in entries]))

    L = len(wt_cds)
    sub_str = ""
    for _, seq in entries:
        for cand_s in (seq, _revcomp(seq)):
            for i in range(len(cand_s) - L + 1):
                win = cand_s[i : i + L]
                diffs = [k for k in range(0, L, 3) if win[k : k + 3] != wt_cds[k : k + 3]]
                if len(diffs) == 1:
                    k = diffs[0]
                    aa_wt = CODON[wt_cds[k : k + 3]]
                    aa_mut = CODON[win[k : k + 3]]
                    if aa_wt != aa_mut:
                        sub_str = f"{aa_wt}{(k // 3) + 1}{aa_mut}"
                        break
            if sub_str:
                break
        if sub_str:
            break

    return {
        "shared_plasmid_family_tag": best_tag,
        "median_plasmid_copy_number": med_cn,
        "amino_acid_substitution": sub_str,
    }
''',
}


def _pack_single_artifact_tar(dst_tar: Path, container_path: str, code: str) -> None:
    rel = container_path.lstrip("/")
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / Path(rel).name
        fp.write_text(code, encoding="utf-8")
        with tarfile.open(dst_tar, "w") as tf:
            tf.add(fp, arcname=rel)


def test_holdout_suite_preregistration_and_airgap_invariants() -> None:
    tasks = build_holdout_ood_12_tasks()
    assert len(tasks) == 12
    assert sum(1 for t in tasks if t.wave == "wave_1") == 6
    assert sum(1 for t in tasks if t.wave == "wave_2") == 6
    sha = compute_holdout_suite_preregistration_sha256()
    assert len(sha) == 64

    waves = holdout_tasks_as_frontier_waves()
    assert len(waves["wave_1"]) == 6 and len(waves["wave_2"]) == 6
    for w_list in waves.values():
        for ft in w_list:
            assert ft.benchmark == "Hegel-OOD-Holdout-12"
            assert not hasattr(ft, "private_verifier_py")
            assert "9202" not in (ft.holdout_public_setup_py or "")


def test_all_12_holdout_tasks_solvable_and_clean_in_sandbox() -> None:
    tasks = build_holdout_ood_12_tasks()
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # 1. Verify empty submission fails strictly
        empty_tar = td_p / "empty.tar"
        with tarfile.open(empty_tar, "w"):
            pass
        neg_rep = run_holdout_private_verifier(tasks[0], empty_tar)
        assert neg_rep["strict_pass"] is False

        # 2. Verify all 12 reference mathematical solvers pass 12/12 inside `hegel_clean_scipy_sandbox:v1`!
        for t in tasks:
            tar_fp = td_p / f"{t.task_id}.tar"
            _pack_single_artifact_tar(tar_fp, t.declared_artifacts[0], REFERENCE_SOLVERS[t.task_id])
            rep = run_holdout_private_verifier(t, tar_fp)
            assert rep["strict_pass"] is True, f"Reference solver failed `{t.task_id}`: {rep}"
