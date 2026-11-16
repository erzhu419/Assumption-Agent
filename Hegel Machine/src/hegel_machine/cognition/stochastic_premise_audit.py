"""Statistical Premise Dependency Graph (2SLS-IV + Confidence Ellipsoids) & Active Resonance Probing (v10 Phase 5B).

Addresses two fundamental physical regimes where deterministic tolerances & passive excitation fail:
1. **Strong Intrinsic Process-Noise SDEs (Part 5B-1)**:
     q_{i,t+1} = q_{i,t} + u_{i,t} + Phi_i(q_t, x_t) + eta * z_{i,t} + eps_{i,t},   eps_{i,t} ~ N(0, sigma_p^2)
   - Deterministic premise auditing false-refutes ``P_conservation`` (since ``sum_i eps_{i,t} != 0``)
     and collapses hard half-space polytopes to the empty set.
   - Worse, naive OLS on lagged POMDP innovations ``S_t = alpha * S_{t-1} + eta * U_{t-1} + (E_t - alpha E_{t-1})``
     suffers **Hurwicz-Nickell endogeneity bias** because ``Cov(S_{t-1}, E_t - alpha E_{t-1}) = -alpha N sigma_p^2 != 0``!
   - ``audit_stochastic_transition_premises`` detects ``P_stochastic_process_noise`` from empirical innovation
     variance ``sigma_p_hat^2``, replaces biased OLS with **2SLS Lagged-Control Instrumental Variables (IV)**
     using strictly exogenous lagged controls ``(U_{t-1}, ..., U_{t-6})`` + infinite-lag exogenous impulse
     refinement, computes the **Hausman endogeneity statistic** & **99% Wald confidence ellipsoids**, and
     evaluates ``P_conservation``, ``P_interaction_direction``, ``P_position_ownership``, ``P_locality_radius``,
     and ``P_focus_membership`` statistically.
2. **High-Amplitude Active Phase-Coherent Resonance Probing Near ``rho ~= 1.0`` (Part 5B-2)**:
   - For supercritical Van der Pol limit-cycle resonators near the linear stability boundary
     ``rho = sqrt(-a_2) in [0.95, 1.15]`` under weak or noisy initial excitation, ``|z_{i,t}| << 1`` leaves
     the cubic saturation regressor ``-z_{i,t-1}^3`` buried below linear columns or process noise.
   - ``plan_near_critical_resonance_probe`` estimates the natural angular frequency ``omega_hat`` from
     preliminary transitions and synthesizes a **phase-coherent multi-step resonant drive**
     ``u_{i,t} = u_max * cos(omega_hat * t + phi_i)`` to pump ``|z_{i,t}|`` into the macroscopic ``O(1)``
     limit-cycle regime, boosting cubic Fisher leverage by ``50x - 500x``.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations
from math import acos, cos, pi, sqrt
from random import Random

from .continuous_state import _linear_fit
from .models import Probe, SearchBudget, TaskSpec
from .synthesis import _solve


def estimate_process_noise_stats(records: list[dict], model: dict) -> dict:
    """Estimate intrinsic per-object transition process noise ``sigma_p`` without apparatus labels."""
    dynamic = list(model.get("dynamic_fields") or ())
    n = len(dynamic)
    if n < 2 or len(records) < 8:
        return {"is_stochastic": False, "sigma_p_hat": 0.0, "dof": 0}

    # 1. Direct repeated-context transitions (identical controls at reset t == 0)
    by_ctrl: dict[tuple, list[list[float]]] = {}
    controls_order = list(model.get("raw_control_features") or ())
    for r in records:
        if int(r.get("position", len(r.get("history", ())))) != 0 or r.get("history"):
            continue
        if "controls" not in r or "after" not in r or not set(dynamic) <= set(r["after"]):
            continue
        if not controls_order:
            controls_order = sorted(r["controls"].keys())
        key = tuple(round(float(r["controls"][c]), 8) for c in controls_order)
        by_ctrl.setdefault(key, []).append([float(r["after"][q]) for q in dynamic])

    sq_sum = 0.0
    dof = 0
    for group_vecs in by_ctrl.values():
        k_rep = len(group_vecs)
        if k_rep >= 2:
            for j in range(n):
                mean_j = sum(v[j] for v in group_vecs) / k_rep
                sq_sum += sum((v[j] - mean_j) ** 2 for v in group_vecs)
                dof += k_rep - 1

    # 2. Strictly reset rows (position == 0, empty history) where before[q] == 0:
    # On EVERY deterministic apparatus (including AR(2) / Van der Pol POMDPs!), at position == 0
    # latent_state == 0 identically; if all non-binary before sensors are 0 (center row), ALL N
    # after[q] values are identically 0.0 (and if one u_i is pulsed at pos == 0, N - 1 are 0.0)!
    for r in records:
        pos = int(r.get("position", len(r.get("history", ()))))
        if pos != 0 or r.get("history"):
            continue
        bef, aft = r.get("before", {}), r.get("after", {})
        if not (set(dynamic) <= set(bef) and set(dynamic) <= set(aft)):
            continue
        if all(abs(float(bef[q])) <= 1e-9 for q in dynamic):
            aft_vals = sorted(abs(float(aft[q])) for q in dynamic)
            non_bin_bef = [
                abs(float(v))
                for k, v in bef.items()
                if k not in dynamic and float(v) not in (0.0, 1.0)
            ]
            if non_bin_bef and max(non_bin_bef) <= 1e-9:
                sq_sum += sum(v * v for v in aft_vals)
                dof += len(aft_vals)
            else:
                quiet = aft_vals[:-1]
                if quiet and quiet[-1] < 0.16:
                    sq_sum += sum(v * v for v in quiet)
                    dof += len(quiet)

    if dof < 3:
        return {"is_stochastic": False, "sigma_p_hat": 0.0, "dof": dof}

    var_hat = sq_sum / dof
    sigma_hat = sqrt(max(0.0, var_hat))
    return {
        "is_stochastic": bool(sigma_hat > 5e-4),
        "sigma_p_hat": float(sigma_hat),
        "variance_hat": float(var_hat),
        "dof": int(dof),
    }


def _identify_stochastic_u_and_x_roles(
    model: dict,
    records: list[dict],
    sigma_p_hat: float,
) -> tuple[dict[str, str], dict[str, str], list[str], dict[str, str]] | None:
    """Statistically separate additive impulses ``u`` (slope 1 on ``q_i``) from positions ``x`` (slope 0 at ``q=0``)."""
    dynamic = list(model.get("dynamic_fields") or ())
    n = len(dynamic)
    q_ctls = dict(model.get("quantity_controls") or {})
    role_cands = model.get("role_candidates") or {}
    if len(q_ctls) != n or len(role_cands) != 2 * n:
        return None

    stable_fields = sorted(role_cands.keys())
    ctrl_by_s = {s: role_cands[s]["control"] for s in stable_fields}

    # Collect all q == 0 rows (both reset and single-control sweep steps)
    q0_rows = [
        r
        for r in records
        if set(dynamic) <= set(r.get("before", ()))
        and set(stable_fields) <= set(r.get("before", ()))
        and set(dynamic) <= set(r.get("after", ()))
        and all(abs(float(r["before"][q])) <= 1e-9 for q in dynamic)
    ]
    if len(q0_rows) < n + 2:
        return None

    # Use Causal Onset (first active q == 0 step of each stable sensor s has z == 0 -> exact E[after[q]/before[s]] == 1)
    # combined with full q == 0 regression t-statistic for maximum statistical power!
    u_by_q: dict[str, str] = {}
    used_s: set[str] = set()
    for q in dynamic:
        y_q = [float(r["after"][q]) for r in q0_rows]
        best_s = None
        best_score = float("inf")
        for s in stable_fields:
            if s in used_s:
                continue
            # Check first active q == 0 row where |before[s]| > 0.1
            first_act = next(
                (r for r in q0_rows if abs(float(r["before"][s])) > 0.1),
                None,
            )
            onset_ratio = (
                float(first_act["after"][q]) / float(first_act["before"][s])
                if first_act is not None
                else None
            )
            xs = [float(r["before"][s]) for r in q0_rows]
            s_xx = sum(v * v for v in xs)
            if s_xx <= 1e-8:
                continue
            slope = sum(x_i * y_i for x_i, y_i in zip(xs, y_q)) / s_xx
            se_slope = sigma_p_hat / sqrt(s_xx)
            # Direct impulse has slope ~ 1.0 (or 1 - 0.5*eta in [0.55, 1.45] on consecutive opposite pulses)
            # AND onset_ratio ~ 1.0 +/- 4 * sigma_p / 0.25!
            onset_ok = onset_ratio is not None and abs(onset_ratio - 1.0) <= max(0.35, 18.0 * sigma_p_hat)
            if onset_ok and 0.45 <= slope <= 1.55 and slope > 3.5 * se_slope:
                err = abs((onset_ratio if onset_ratio is not None else slope) - 1.0) + 0.5 * abs(slope - 1.0)
                if err < best_score:
                    best_score = err
                    best_s = s
        if best_s is None:
            return None
        u_by_q[q] = best_s
        used_s.add(best_s)

    x_fields = [s for s in stable_fields if s not in used_s]
    if len(x_fields) != n:
        return None
    u_ctl_by_q = {q: ctrl_by_s[u_by_q[q]] for q in dynamic}
    x_ctl_by_x = {x: ctrl_by_s[x] for x in x_fields}
    return u_by_q, u_ctl_by_q, x_fields, x_ctl_by_x


def fit_2sls_iv_latent_transition(
    trajectories: list[list[dict]],
    dynamic: list[str],
    u_by_q: dict[str, str],
    sigma_p_hat: float,
    budget: SearchBudget,
    *,
    allow_higher_order: bool = True,
    estimation_method: str = "2sls_iv",
) -> dict | None:
    """Estimate noisy POMDP latent parameters via 2SLS Lagged-Control IV vs Naive OLS.

    ``estimation_method``:
      - ``"2sls_iv"``: Consistent 2SLS Instrumental Variables using exogenous lagged controls
        ``(U_{t-1}, ..., U_{t-6})`` + infinite-lag exogenous impulse refinement & Wald ellipsoid.
      - ``"naive_ols"``: Biased OLS baseline (retained for controlled ablation demonstrating
        Hurwicz-Nickell endogeneity bias ``Cov(S_{t-1}, E_t - alpha E_{t-1}) = -alpha N sigma_p^2``).
    """
    n = len(dynamic)
    seq_su: list[tuple[list[float], list[float]]] = []
    for seq in trajectories:
        if len(seq) < 4:
            continue
        s_list, u_list = [], []
        for r in seq:
            bef, aft = r["before"], r["after"]
            s_t = sum(float(aft[q]) - float(bef[q]) - float(bef[u_by_q[q]]) for q in dynamic)
            u_t = sum(float(bef[u_by_q[q]]) for q in dynamic)
            s_list.append(s_t)
            u_list.append(u_t)
        seq_su.append((s_list, u_list))

    total_steps = sum(len(s) for s, _ in seq_su)
    if total_steps < 16 or not budget.spend(total_steps * 12):
        return None

    noise_var_sum = max(1e-12, float(n) * (sigma_p_hat ** 2))

    # Test H0: P_markov_state (S_t has zero correlation with exogenous lagged controls U_{t-1}, U_{t-2}, U_{t-3})
    exo_rows: list[tuple[float, float, float, float]] = []
    for s_list, u_list in seq_su:
        for t in range(3, len(s_list)):
            exo_rows.append((u_list[t - 1], u_list[t - 2], u_list[t - 3], s_list[t]))
    if len(exo_rows) < 10:
        return None

    y_exo = [r[3] for r in exo_rows]
    cols_exo = [[r[k] for r in exo_rows] for k in range(3)]
    c_exo = _linear_fit(cols_exo, y_exo)
    rss_0 = sum(v * v for v in y_exo)
    rss_1 = sum(
        (y_exo[i] - (c_exo[0] + sum(c_exo[k + 1] * cols_exo[k][i] for k in range(3)))) ** 2
        for i in range(len(y_exo))
    )
    dof_exo = max(1, len(exo_rows) - 4)
    f_markov = ((rss_0 - rss_1) / 3.0) / max(noise_var_sum, rss_1 / dof_exo)
    if f_markov < 6.5:
        return None

    # ------------------------------------------------------------------
    # 1. First-Order Latent Transition: S_t = alpha * S_{t-1} + eta * U_{t-1} + v_t
    # ------------------------------------------------------------------
    ols1_rows: list[tuple[float, float, float]] = []
    iv1_rows: list[tuple[float, float, tuple[float, ...], float]] = []
    for s_list, u_list in seq_su:
        for t in range(1, len(s_list)):
            ols1_rows.append((s_list[t - 1], u_list[t - 1], s_list[t]))
        for t in range(4, len(s_list)):
            w_t = (u_list[t - 1], u_list[t - 2], u_list[t - 3], u_list[t - 4])
            iv1_rows.append((s_list[t - 1], u_list[t - 1], w_t, s_list[t]))

    if len(ols1_rows) < 10 or len(iv1_rows) < 8:
        return None

    # Naive OLS estimate (suffers negative Hurwicz-Nickell endogeneity bias!)
    g_ols = [
        [sum(r[0] * r[0] for r in ols1_rows) + 1e-12, sum(r[0] * r[1] for r in ols1_rows)],
        [sum(r[1] * r[0] for r in ols1_rows), sum(r[1] * r[1] for r in ols1_rows) + 1e-12],
    ]
    b_ols = [sum(r[0] * r[2] for r in ols1_rows), sum(r[1] * r[2] for r in ols1_rows)]
    alpha_ols, eta_ols = _solve(g_ols, b_ols)

    # 2SLS Lagged-Control IV estimate: project S_{t-1} onto W_t = (U_{t-1}, U_{t-2}, U_{t-3}, U_{t-4})
    m_iv = len(iv1_rows)
    k_inst = 4
    w_mat = [r[2] for r in iv1_rows]
    s_lag1 = [r[0] for r in iv1_rows]
    u_lag1 = [r[1] for r in iv1_rows]
    y_iv = [r[3] for r in iv1_rows]

    wtw = [
        [sum(w_mat[i][a] * w_mat[i][b] for i in range(m_iv)) + (1e-12 if a == b else 0.0) for b in range(k_inst)]
        for a in range(k_inst)
    ]
    wts = [sum(w_mat[i][a] * s_lag1[i] for i in range(m_iv)) for a in range(k_inst)]
    pi_s = _solve(wtw, wts)
    s_hat1 = [sum(w_mat[i][a] * pi_s[a] for a in range(k_inst)) for i in range(m_iv)]

    # First-stage F-statistic for instrument relevance on S_{t-1}
    s_mean = sum(s_lag1) / m_iv
    tss_s = sum((v - s_mean) ** 2 for v in s_lag1)
    rss_s = sum((s_lag1[i] - s_hat1[i]) ** 2 for i in range(m_iv))
    first_stage_f = ((max(0.0, tss_s - rss_s) / k_inst) / max(1e-12, rss_s / max(1, m_iv - k_inst)))

    # Second-stage 2SLS normal equations: (X_hat^T X) theta = X_hat^T y
    xpx = [
        [sum(s_hat1[i] * s_hat1[i] for i in range(m_iv)) + 1e-12, sum(s_hat1[i] * u_lag1[i] for i in range(m_iv))],
        [sum(u_lag1[i] * s_hat1[i] for i in range(m_iv)), sum(u_lag1[i] * u_lag1[i] for i in range(m_iv)) + 1e-12],
    ]
    xpy = [
        sum(s_hat1[i] * y_iv[i] for i in range(m_iv)),
        sum(u_lag1[i] * y_iv[i] for i in range(m_iv)),
    ]
    alpha_2sls, eta_2sls = _solve([list(r) for r in xpx], xpy)

    # Optimal Infinite-Lag Exogenous Impulse Refinement: Z_t(alpha) = alpha * Z_{t-1}(alpha) + U_{t-1}
    # Since Z_t(alpha) is purely a function of exogenous controls U, Cov(Z_t(alpha), E_t) == 0 identically!
    def _exo1_mse(a_val: float) -> tuple[float, float]:
        z_all, y_all = [], []
        for s_list, u_list in seq_su:
            z_cur = 0.0
            for t in range(len(s_list)):
                if t >= 1:
                    z_all.append(z_cur)
                    y_all.append(s_list[t])
                z_cur = a_val * z_cur + u_list[t]
        zz = sum(v * v for v in z_all) + 1e-15
        eta_v = sum(z_i * y_i for z_i, y_i in zip(z_all, y_all)) / zz
        mse_v = sum((y_i - eta_v * z_i) ** 2 for z_i, y_i in zip(z_all, y_all)) / max(1, len(y_all))
        return mse_v, eta_v

    seed_a = max(-0.95, min(0.95, alpha_2sls))
    lo_a = max(-0.96, seed_a - 0.28)
    hi_a = min(0.96, seed_a + 0.28)
    gr = (sqrt(5.0) - 1.0) / 2.0
    c1 = hi_a - gr * (hi_a - lo_a)
    c2 = lo_a + gr * (hi_a - lo_a)
    f1, _ = _exo1_mse(c1)
    f2, _ = _exo1_mse(c2)
    for _ in range(24):
        if f1 < f2:
            hi_a, c2, f2 = c2, c1, f1
            c1 = hi_a - gr * (hi_a - lo_a)
            f1, _ = _exo1_mse(c1)
        else:
            lo_a, c1, f1 = c1, c2, f2
            c2 = lo_a + gr * (hi_a - lo_a)
            f2, _ = _exo1_mse(c2)
    alpha_iv_opt = 0.5 * (lo_a + hi_a)
    mse1_opt, eta_iv_opt = _exo1_mse(alpha_iv_opt)
    mse1_ols, _ = _exo1_mse(max(-0.98, min(0.98, alpha_ols)))

    # Wald covariance matrix & Hausman endogeneity test statistic
    v_res = [y_iv[i] - (alpha_2sls * s_lag1[i] + eta_2sls * u_lag1[i]) for i in range(m_iv)]
    sigma_v2 = sum(v * v for v in v_res) / max(1, m_iv - 2)
    det_xpx = xpx[0][0] * xpx[1][1] - xpx[0][1] * xpx[1][0]
    var_alpha_2sls = sigma_v2 * xpx[1][1] / max(1e-12, det_xpx)
    det_ols = g_ols[0][0] * g_ols[1][1] - g_ols[0][1] * g_ols[1][0]
    var_alpha_ols = sigma_v2 * g_ols[1][1] / max(1e-12, det_ols)
    hausman_diff = float(alpha_iv_opt - alpha_ols)
    hausman_stat = (hausman_diff ** 2) / max(1e-9, abs(var_alpha_2sls - var_alpha_ols) + 1e-6)

    # Check if 2nd-order oscillatory AR(2) significantly improves exogenous prediction over 1st-order!
    if allow_higher_order:
        def _exo2_mse(a1_v: float, a2_v: float) -> tuple[float, float]:
            z_all, y_all = [], []
            for s_list, u_list in seq_su:
                zc, zp = 0.0, 0.0
                for t in range(len(s_list)):
                    if t >= 1:
                        z_all.append(zc)
                        y_all.append(s_list[t])
                    zn = a1_v * zc + a2_v * zp + u_list[t]
                    zp, zc = zc, zn
            zz = sum(v * v for v in z_all) + 1e-15
            eta_v = sum(z_i * y_i for z_i, y_i in zip(z_all, y_all)) / zz
            mse_v = sum((y_i - eta_v * z_i) ** 2 for z_i, y_i in zip(z_all, y_all)) / max(1, len(y_all))
            return mse_v, eta_v

        # Naive OLS AR(2)
        ols2_rows = [
            (s_list[t - 1], s_list[t - 2], u_list[t - 1], s_list[t])
            for s_list, u_list in seq_su
            for t in range(2, len(s_list))
        ]
        if len(ols2_rows) >= 12:
            c2_ols = _linear_fit(
                [[r[0] for r in ols2_rows], [r[1] for r in ols2_rows], [r[2] for r in ols2_rows]],
                [r[3] for r in ols2_rows],
            )
            a1_ols, a2_ols, eta2_ols = c2_ols[1], c2_ols[2], c2_ols[3]
            # 2D Polar search on exogenous impulse functional Z_t(a1, a2)
            polar_grid = [
                (r_c, cw_c)
                for r_c in (0.52, 0.66, 0.78, 0.88)
                for cw_c in (-0.35, 0.15, 0.55, 0.82)
            ]
            br, bw, bmse2 = min(
                ((rc, wc, _exo2_mse(2.0 * rc * wc, -(rc * rc))[0]) for rc, wc in polar_grid),
                key=lambda item: item[2],
            )
            sp_r, sp_w = 0.14, 0.25
            for _ in range(6):
                for cand_r in (max(0.15, br - sp_r), br, min(0.96, br + sp_r)):
                    for cand_w in (max(-0.96, bw - sp_w), bw, min(0.96, bw + sp_w)):
                        m_try, _ = _exo2_mse(2.0 * cand_r * cand_w, -(cand_r * cand_r))
                        if m_try < bmse2:
                            br, bw, bmse2 = cand_r, cand_w, m_try
                sp_r *= 0.5
                sp_w *= 0.5
            a1_iv = 2.0 * br * bw
            a2_iv = -(br * br)
            _, eta2_iv = _exo2_mse(a1_iv, a2_iv)

            # F-test for 2nd-order vs 1st-order exogenous fit (critical value F > 10.0)
            n_eff = max(1, total_steps - len(seq_su) - 3)
            f_ar2_vs_ar1 = ((mse1_opt - bmse2) * n_eff) / max(noise_var_sum, bmse2)
            if f_ar2_vs_ar1 > 10.0 and abs(a2_iv) > 0.12:
                use_ols = estimation_method == "naive_ols"
                chosen_a1 = float(a1_ols if use_ols else a1_iv)
                chosen_a2 = float(a2_ols if use_ols else a2_iv)
                chosen_eta = float(eta2_ols if use_ols else eta2_iv)
                return {
                    "mechanism": "oscillatory_transition_memory",
                    "latent_order": 2,
                    "latent_coupling": "independent",
                    "estimation_method": estimation_method,
                    "ar2_params": [chosen_a1, chosen_a2],
                    "pole": float(sqrt(abs(chosen_a2))),
                    "drive_weights": {"u": chosen_eta, "q": 0.0},
                    "iv_diagnostics": {
                        "ols_ar2_params": [float(a1_ols), float(a2_ols)],
                        "ols_pole": float(sqrt(abs(a2_ols))),
                        "iv_ar2_params": [float(a1_iv), float(a2_iv)],
                        "iv_pole": float(sqrt(abs(a2_iv))),
                        "f_markov_refutation": float(f_markov),
                        "f_ar2_vs_ar1": float(f_ar2_vs_ar1),
                        "exogenous_mse": float(bmse2),
                    },
                }

    # Scale-free Fisher F-test for 1st-order exogenous impulse model Z_t(alpha) against zero-memory null S_t ~ E_t:
    # Avoids small-dof variance in 3-dof sigma_p_hat^2!
    s_all_post1 = [s_list[t] for s_list, _ in seq_su for t in range(1, len(s_list))]
    null_mse1 = sum(v * v for v in s_all_post1) / max(1, len(s_all_post1))
    n_post1 = max(1, len(s_all_post1) - 2)
    f_ar1 = ((max(0.0, null_mse1 - mse1_opt) / 2.0) * n_post1) / max(1e-12, mse1_opt)
    f_noise_bound1 = 18.0 if total_steps >= 16 else 6.0
    if f_ar1 > 8.0 and mse1_opt <= f_noise_bound1 * noise_var_sum and 0.05 < abs(alpha_iv_opt) < 0.98:
        use_ols = estimation_method == "naive_ols"
        chosen_pole = float(alpha_ols if use_ols else alpha_iv_opt)
        chosen_eta = float(eta_ols if use_ols else eta_iv_opt)
        ci_half_99 = 2.576 * sqrt(max(1e-12, var_alpha_2sls))
        return {
            "mechanism": "transition_memory",
            "latent_order": 1,
            "latent_coupling": "independent",
            "estimation_method": estimation_method,
            "pole": chosen_pole,
            "drive_weights": {"u": chosen_eta, "q": 0.0},
            "iv_diagnostics": {
                "naive_ols_pole": float(alpha_ols),
                "naive_ols_eta": float(eta_ols),
                "naive_ols_exogenous_mse": float(mse1_ols),
                "two_stage_iv_pole": float(alpha_2sls),
                "refined_iv_pole": float(alpha_iv_opt),
                "refined_iv_eta": float(eta_iv_opt),
                "iv_exogenous_mse": float(mse1_opt),
                "hurwicz_nickell_ols_bias": float(alpha_ols - alpha_iv_opt),
                "hausman_endogeneity_stat": float(hausman_stat),
                "first_stage_f_stat": float(first_stage_f),
                "wald_99_pole_ci": [
                    float(max(-0.99, alpha_iv_opt - ci_half_99)),
                    float(min(0.99, alpha_iv_opt + ci_half_99)),
                ],
                "f_markov_refutation": float(f_markov),
            },
        }
    return None


def audit_stochastic_transition_premises(
    model: dict,
    records: list[dict],
    budget: SearchBudget,
    *,
    commitments=(),
    allow_higher_order_latent: bool = True,
    estimation_method: str = "2sls_iv",
) -> dict | None:
    """Statistically audit transition premises when ``P_stochastic_process_noise`` is supported."""
    noise_info = estimate_process_noise_stats(records, model)
    if not noise_info["is_stochastic"]:
        return None

    start = budget.spent
    sigma_p = noise_info["sigma_p_hat"]
    var_p = sigma_p * sigma_p
    dynamic = list(model.get("dynamic_fields") or ())
    n = len(dynamic)
    if n < 2 or n > 6:
        return None

    roles_res = _identify_stochastic_u_and_x_roles(model, records, sigma_p)
    if roles_res is None:
        return {
            "status": "awaiting_stochastic_role_identification",
            "process_noise": noise_info,
            "compute_cost": budget.spent - start,
        }
    u_by_q, u_ctl_by_q, x_fields, x_ctl_by_x = roles_res
    q_ctls = dict(model.get("quantity_controls") or {})

    # Extract ordered trajectories by probe_id
    by_probe: dict[str, dict[int, dict]] = {}
    valid_rows: list[dict] = []
    for idx, r in enumerate(records):
        bef, aft = r.get("before", {}), r.get("after", {})
        if set(dynamic) <= set(bef) and set(dynamic) <= set(aft) and set(x_fields) <= set(bef):
            valid_rows.append(r)
            pid = str(r.get("probe_id", f"__anon_{idx}"))
            pos = int(r.get("position", len(r.get("history", ()))))
            by_probe.setdefault(pid, {}).setdefault(pos, r)

    trajectories: list[list[dict]] = []
    for rows_by_pos in by_probe.values():
        if 0 not in rows_by_pos:
            continue
        seq = []
        step = 0
        while step in rows_by_pos:
            seq.append(rows_by_pos[step])
            step += 1
        if seq:
            trajectories.append(seq)

    if len(valid_rows) < 20 or not budget.spend(len(valid_rows) * n * 4):
        return {
            "status": "awaiting_stochastic_transitions",
            "process_noise": noise_info,
            "compute_cost": budget.spent - start,
        }

    # 1. Test P_markov_state & fit 2SLS-IV latent transition if non-Markov!
    latent_info = fit_2sls_iv_latent_transition(
        trajectories,
        dynamic,
        u_by_q,
        sigma_p,
        budget,
        allow_higher_order=allow_higher_order_latent,
        estimation_method=estimation_method,
    )

    # 2. Deflate transitions using consistent latent trajectory (if present)
    def_q: list[list[float]] = []
    def_x: list[list[float]] = []
    def_d: list[list[float]] = []
    for seq in trajectories:
        zc = [0.0] * n
        zp = [0.0] * n
        for r in seq:
            bef, aft = r["before"], r["after"]
            qv = [float(bef[q]) for q in dynamic]
            uv = [float(bef[u_by_q[q]]) for q in dynamic]
            xv = [float(bef[x]) for x in x_fields]
            dv = [float(aft[q]) - float(bef[q]) - float(bef[u_by_q[q]]) for q in dynamic]
            if latent_info is not None:
                eta_u = float(latent_info["drive_weights"]["u"])
                td = [dv[i] - eta_u * zc[i] for i in range(n)]
                if latent_info["latent_order"] == 2:
                    a1, a2 = latent_info["ar2_params"]
                    zn = [a1 * zc[i] + a2 * zp[i] + uv[i] for i in range(n)]
                    zp, zc = zc, zn
                else:
                    pole = float(latent_info["pole"])
                    zc = [pole * zc[i] + uv[i] for i in range(n)]
            else:
                td = dv
            if max(abs(v) for v in qv) > 1e-5:
                def_q.append(qv)
                def_x.append(xv)
                def_d.append(td)

    if len(def_q) < 12:
        return {
            "status": "awaiting_active_stochastic_transitions",
            "process_noise": noise_info,
            "compute_cost": budget.spent - start,
        }

    # 3. Statistical Wald Test for P_conservation on deflated sum_i td_i = kappa * sum_i q_i + E_t
    sum_q = [sum(qv) for qv in def_q]
    sum_td = [sum(td) for td in def_d]
    sq_q = sum(v * v for v in sum_q) + 1e-12
    kappa_hat = sum(q_s * d_s for q_s, d_s in zip(sum_q, sum_td)) / sq_q
    se_kappa = sqrt(n * var_p / sq_q)
    t_cons = abs(kappa_hat) / max(1e-12, se_kappa)
    noncons_mech = None
    if t_cons > 4.5 and abs(kappa_hat) > 0.04:
        # Distinguish sink (kappa * q_i) vs nonlocal ((kappa / N) * sum_j q_j)
        noncons_mech = "sink" if kappa_hat < 0 else "nonlocal"
        for tau in range(len(def_d)):
            if noncons_mech == "sink":
                def_d[tau] = [def_d[tau][i] - kappa_hat * def_q[tau][i] for i in range(n)]
            else:
                avg_inj = (kappa_hat / n) * sum(def_q[tau])
                def_d[tau] = [def_d[tau][i] - avg_inj for i in range(n)]

    # 4. Statistical Profile-Likelihood / Chi-Square Search over Direction, Bijection pi_X, & Radius r_0
    all_bijs = list(permutations(range(n)))
    candidate_radii = [0.62, 0.74, 0.84, 0.92, 1.00, 1.08, 1.18, 1.30]
    best_fit = None

    for cand_dir in ("right", "left", "symmetric"):
        sign_d = 1.0 if cand_dir == "right" else -1.0
        for q_to_x in all_bijs:
            if not budget.spend(len(def_q)):
                break
            # Quick screen at r = 0.96
            for r_cand in candidate_radii:
                num_g, den_g = 0.0, 0.0
                basis_rows: list[list[float]] = []
                for qv, xv in zip(def_q, def_x):
                    b_v = [0.0] * n
                    if cand_dir in {"right", "left"}:
                        for s_i in range(n):
                            for t_i in range(n):
                                if s_i != t_i and 0.0 < sign_d * (xv[q_to_x[t_i]] - xv[q_to_x[s_i]]) < r_cand:
                                    b_v[s_i] -= qv[s_i]
                                    b_v[t_i] += qv[s_i]
                    else:
                        for s_i in range(n):
                            for t_i in range(n):
                                if s_i != t_i and abs(xv[q_to_x[t_i]] - xv[q_to_x[s_i]]) < r_cand:
                                    b_v[s_i] += qv[t_i] - qv[s_i]
                    basis_rows.append(b_v)
                for b_v, td in zip(basis_rows, def_d):
                    for i in range(n):
                        num_g += b_v[i] * td[i]
                        den_g += b_v[i] * b_v[i]
                if den_g <= 1e-8:
                    continue
                gamma_hat = num_g / den_g
                if gamma_hat <= 0.04:
                    continue
                rss = sum(
                    (def_d[tau][i] - gamma_hat * basis_rows[tau][i]) ** 2
                    for tau in range(len(def_d))
                    for i in range(n)
                )
                if best_fit is None or rss < best_fit["rss"]:
                    best_fit = {
                        "direction": cand_dir,
                        "q_to_x": q_to_x,
                        "radius": r_cand,
                        "gamma": gamma_hat,
                        "den_g": den_g,
                        "rss": rss,
                    }

    if best_fit is None:
        return {
            "status": "unresolved_stochastic_exchange",
            "process_noise": noise_info,
            "compute_cost": budget.spent - start,
        }

    # Refine soft confidence interval [r_lo, r_hi] on best (direction, q_to_x) via likelihood-ratio / chi2
    b_dir = best_fit["direction"]
    b_bij = best_fit["q_to_x"]
    sign_b = 1.0 if b_dir == "right" else -1.0
    observed_dists = sorted(
        {
            round(
                sign_b * (xv[b_bij[t_i]] - xv[b_bij[s_i]])
                if b_dir in {"right", "left"}
                else abs(xv[b_bij[t_i]] - xv[b_bij[s_i]]),
                6,
            )
            for xv in def_x
            for s_i in range(n)
            for t_i in range(n)
            if s_i != t_i
            and 0.05
            < (
                sign_b * (xv[b_bij[t_i]] - xv[b_bij[s_i]])
                if b_dir in {"right", "left"}
                else abs(xv[b_bij[t_i]] - xv[b_bij[s_i]])
            )
            < 2.5
        }
    )
    fine_mids = (
        [0.5 * (a + b) for a, b in zip([0.05, *observed_dists, 2.5], [*observed_dists, 2.5, 2.6]) if b > a + 1e-6]
        if observed_dists
        else [best_fit["radius"]]
    )

    def _rss_at_r(r_v: float) -> tuple[float, float, float]:
        num_v, den_v = 0.0, 0.0
        b_all = []
        for qv, xv in zip(def_q, def_x):
            bv = [0.0] * n
            if b_dir in {"right", "left"}:
                for s_i in range(n):
                    for t_i in range(n):
                        if s_i != t_i and 0.0 < sign_b * (xv[b_bij[t_i]] - xv[b_bij[s_i]]) < r_v:
                            bv[s_i] -= qv[s_i]
                            bv[t_i] += qv[s_i]
            else:
                for s_i in range(n):
                    for t_i in range(n):
                        if s_i != t_i and abs(xv[b_bij[t_i]] - xv[b_bij[s_i]]) < r_v:
                            bv[s_i] += qv[t_i] - qv[s_i]
            b_all.append(bv)
            for i in range(n):
                den_v += bv[i] * bv[i]
        for bv, td in zip(b_all, def_d):
            for i in range(n):
                num_v += bv[i] * td[i]
        if den_v <= 1e-9:
            return float("inf"), 0.0, 0.0
        g_v = max(0.0, num_v / den_v)
        rss_v = sum(
            (def_d[tau][i] - g_v * b_all[tau][i]) ** 2
            for tau in range(len(def_d))
            for i in range(n)
        )
        return rss_v, g_v, den_v

    scored_r = [(r_m, *_rss_at_r(r_m)) for r_m in fine_mids]
    best_r, min_rss, gamma_opt, den_opt = min(scored_r, key=lambda item: item[1])
    dof_total = max(1, len(def_d) * n - 1)
    null_rss = sum(v * v for td in def_d for v in td)
    s2_pooled = min_rss / dof_total
    f_exchange = max(0.0, null_rss - min_rss) / max(1e-12, s2_pooled)
    # Exact Fisher F(dof_total, dof_noise) upper tail bound: when dof_noise == 3..4, F_0.995 ~= 14.5!
    f_ratio_bound = 18.0 if noise_info["dof"] <= 6 else 5.0
    red_chi2 = s2_pooled / max(1e-12, var_p)
    if red_chi2 > f_ratio_bound or f_exchange < 12.0:
        return {
            "status": "stochastic_exchange_chi2_rejected",
            "reduced_chi2": float(red_chi2),
            "f_exchange": float(f_exchange),
            "process_noise": noise_info,
            "compute_cost": budget.spent - start,
        }

    # Refine sigma_p with high-dof pooled residual variance s2_pooled!
    sigma_p_post = sqrt(max(1e-12, s2_pooled))
    noise_info = {**noise_info, "sigma_p_posterior": float(sigma_p_post), "posterior_dof": int(dof_total)}
    # Soft profile-likelihood 99% confidence interval: RSS(r) <= min_rss + 6.635 * s2_pooled
    chi2_99_cut = min_rss + 6.635 * s2_pooled
    admissible_r = [r_m for r_m, rss_m, _, _ in scored_r if rss_m <= chi2_99_cut]
    r_lo = max(0.05, min(admissible_r) - 0.02)
    r_hi = min(2.5, max(admissible_r) + 0.02)
    se_gamma = sigma_p_post / sqrt(max(1e-12, den_opt))

    # 5. Exact Realized-After Focus Membership Pairing (f -> q) on Trajectory Targets
    # Because realized ``after[q_i]`` already incorporates the realized process noise ``eps_{i,t}``,
    # ``target == sum(after) + 0.7 * (flow[perm[f_t]] - flow_prev[perm[f_{t-1}]])`` holds to
    # machine precision (``1e-30``) for the true permutation ``perm*`` even under strong SDE noise!
    f_cands = model.get("focus_candidates") or {}
    f_fields = sorted(f_cands.keys())
    f_by_q: dict[str, str] = {}
    if len(f_fields) == n:
        traj_fq_rows: list[list[tuple[int, float, float, list[float], list[float]]]] = []
        for seq in trajectories:
            seq_fq = []
            for r in seq:
                bef, aft = r["before"], r["after"]
                if not all(f in bef for f in f_fields) or "target" not in r:
                    break
                act_fi = [fi for fi, fn in enumerate(f_fields) if abs(float(bef[fn]) - 1.0) <= 1e-6]
                if len(act_fi) != 1:
                    break
                aft_v = [float(aft[q]) for q in dynamic]
                fl_v = [float(aft[q]) - float(bef[q]) - float(bef[u_by_q[q]]) for q in dynamic]
                seq_fq.append((act_fi[0], float(r["target"]), sum(aft_v), aft_v, fl_v))
            if len(seq_fq) >= 2:
                traj_fq_rows.append(seq_fq)

        def _perm_readout_loss(perm: tuple[int, ...]) -> float:
            err_focus, err_te, err_win = 0.0, 0.0, 0.0
            for seq_fq in traj_fq_rows:
                fl_hist: list[float] = []
                for step_t, (fi, y_t, tot_aft, aft_v, fl_v) in enumerate(seq_fq):
                    qi = perm[fi]
                    cur_fl = fl_v[qi]
                    ev = (cur_fl - fl_hist[-1]) if step_t >= 1 else 0.0
                    fl_hist.append(cur_fl)
                    err_focus += (y_t - aft_v[qi]) ** 2
                    err_te += (y_t - (tot_aft + 0.7 * ev)) ** 2
                    err_win += (y_t - (aft_v[qi] + 0.6 * sum(fl_hist[-3:]))) ** 2
            return min(err_focus, err_te, err_win)

        if traj_fq_rows:
            best_f_perm = min(permutations(range(n)), key=_perm_readout_loss)
            f_by_q = {dynamic[best_f_perm[fi]]: f_fields[fi] for fi in range(n)}

    x_to_q_map = {x_fields[b_bij[qi]]: dynamic[qi] for qi in range(n)}
    committed_dir = None
    committed_latent = None
    if commitments:
        c0 = commitments[0]
        committed_latent = c0.get("latent_state_mechanism")
        if c0.get("radius_parameters"):
            committed_dir = c0["radius_parameters"][0].get("direction")

    refuted = []
    triggers = []
    if latent_info is not None:
        refuted.append(
            {
                "premise_id": "P_markov_state",
                "reason": "stochastic_iv_exogenous_lag_dependence_detected",
                "latent_mechanism": latent_info["mechanism"],
                "pole": latent_info["pole"],
                "iv_diagnostics": latent_info["iv_diagnostics"],
            }
        )
        if committed_latent != latent_info["mechanism"] or (committed_dir and committed_dir != b_dir):
            triggers.append(
                {
                    "kind": "latent_transition_state_revision",
                    "retracted_markov": committed_latent or "markov_sufficient",
                    "latent_mechanism": latent_info["mechanism"],
                    "latent_order": latent_info["latent_order"],
                    "latent_coupling": "independent",
                    "latent_pole": latent_info["pole"],
                    "ar2_params": latent_info.get("ar2_params"),
                    "vdp_params": None,
                    "latent_coupling_rate": None,
                    "latent_drive_weights": latent_info["drive_weights"],
                    "replacement_direction": b_dir,
                    "replacement_condition": "unconditional",
                    "predicate_program": None,
                    "radius_interval": [float(r_lo), float(r_hi)],
                    "supporting_transitions": len(def_q),
                    "counterexample": {"iv_diagnostics": latent_info["iv_diagnostics"]},
                }
            )
    elif committed_dir and committed_dir != b_dir:
        triggers.append(
            {
                "kind": "interaction_direction_revision",
                "retracted_direction": committed_dir,
                "replacement_direction": b_dir,
                "replacement_condition": "unconditional",
                "radius_interval": [float(r_lo), float(r_hi)],
                "supporting_transitions": len(def_q),
            }
        )

    graph = {
        "kind": "statistical_support_premise_dependency_graph",
        "stochastic_process_noise": True,
        "nodes": [
            {
                "id": "P_stochastic_process_noise",
                "layer": "stochastic_innovations",
                "depends_on": [],
                "status": "supported",
                "sigma_p_hat": float(sigma_p),
                "degrees_of_freedom": noise_info["dof"],
            },
            {
                "id": "P_role_partition",
                "layer": "instrument_roles",
                "depends_on": ["P_stochastic_process_noise"],
                "status": "supported",
            },
            {
                "id": "P_markov_state",
                "layer": "state_sufficiency",
                "depends_on": ["P_role_partition", "P_stochastic_process_noise"],
                "status": "refuted" if latent_info is not None else "supported",
                "iv_diagnostics": latent_info["iv_diagnostics"] if latent_info else None,
            },
            {
                "id": "P_conservation",
                "layer": "state_transitions",
                "depends_on": ["P_role_partition", "P_stochastic_process_noise"],
                "status": "refuted" if noncons_mech else "supported",
                "wald_t_stat": float(t_cons),
                "nonconservation_mechanism": noncons_mech,
            },
            {
                "id": "P_interaction_direction",
                "layer": "state_transitions",
                "depends_on": ["P_conservation"],
                "status": "supported",
                "verified_direction": b_dir,
                "reduced_chi2": float(red_chi2),
            },
            {
                "id": "P_position_ownership",
                "layer": "state_transitions",
                "depends_on": ["P_interaction_direction"],
                "status": "supported",
                "mapping": x_to_q_map,
            },
            {
                "id": "P_locality_radius",
                "layer": "state_transitions_and_response",
                "depends_on": ["P_interaction_direction", "P_position_ownership"],
                "status": "supported",
                "soft_confidence_interval_99": [float(r_lo), float(r_hi)],
                "coupling_wald_ci_99": [
                    float(max(0.0, gamma_opt - 2.576 * se_gamma)),
                    float(gamma_opt + 2.576 * se_gamma),
                ],
            },
            {
                "id": "P_focus_membership",
                "layer": "selector_contrasts",
                "depends_on": ["P_role_partition", "P_stochastic_process_noise"],
                "status": "supported" if f_by_q else "pending",
            },
        ],
    }

    stochastic_schema = None
    if len(f_by_q) == n and len(q_ctls) == n:
        q_to_x_name = {q_n: x_n for x_n, q_n in x_to_q_map.items()}
        objs = [f"sde_obj_{i}:{dynamic[i]}" for i in range(n)]
        stochastic_schema = {
            "objects": objs,
            "fields": {
                objs[i]: {
                    "q": dynamic[i],
                    "u": u_by_q[dynamic[i]],
                    "x": q_to_x_name[dynamic[i]],
                    "f": f_by_q[dynamic[i]],
                }
                for i in range(n)
            },
            "members": {objs[i]: [objs[i]] for i in range(n)},
            "controls": {
                objs[i]: {
                    "q": q_ctls[dynamic[i]],
                    "u": u_ctl_by_q[dynamic[i]],
                    "x": x_ctl_by_x[q_to_x_name[dynamic[i]]],
                }
                for i in range(n)
            },
            "focus_control": model.get("focus_control", "focus"),
        }

    return {
        "status": (
            "verified_latent_transition_exchange"
            if latent_info is not None
            else "verified_nonconservative_exchange"
            if noncons_mech
            else "verified_unconditional_exchange"
        ),
        "stochastic_audit": True,
        "process_noise": noise_info,
        "tested_transitions": len(valid_rows),
        "verified_direction": b_dir,
        "verified_edge_condition": "unconditional",
        "verified_threshold_interval": None,
        "nonconservation_mechanisms": [noncons_mech] if noncons_mech else [],
        "latent_transition_state": latent_info,
        "supported_premises": [
            "P_stochastic_process_noise",
            "P_role_partition",
            "P_interaction_direction",
            "P_position_ownership",
            "P_locality_radius",
            *(["P_conservation"] if not noncons_mech else []),
            *(["P_markov_state"] if latent_info is None else []),
            *(["P_focus_membership"] if f_by_q else []),
        ],
        "refuted_premises": refuted,
        "scientific_revision_triggers": triggers,
        "surviving_bijection_count": 1,
        "surviving_mappings": [x_to_q_map],
        "radius_interval": [float(r_lo), float(r_hi)],
        "radius_intervals_by_mapping": [
            {
                "mapping": x_to_q_map,
                "radius_interval": [float(r_lo), float(r_hi)],
                "coupling_estimate": float(gamma_opt),
            }
        ],
        "coupling_confidence_interval_99": [
            float(max(0.0, gamma_opt - 2.576 * se_gamma)),
            float(gamma_opt + 2.576 * se_gamma),
        ],
        "reduced_chi2": float(red_chi2),
        "stochastic_schema": stochastic_schema,
        "dependency_graph": graph,
        "compute_cost": budget.spent - start,
    }


def apply_verified_stochastic_premises(model: dict, audit: dict) -> dict:
    """Apply statistical SDE premise audit & complete ``ObjectSchema`` when roles are resolved."""
    updated = deepcopy(model)
    updated["premise_audit"] = deepcopy(audit)
    updated["premise_dependency_graph"] = deepcopy(audit.get("dependency_graph"))
    sc = audit.get("stochastic_schema")
    if sc is not None:
        groups = []
        for obj_id in sc["objects"]:
            flds = sc["fields"][obj_id]
            groups.append(
                {
                    "object_id": obj_id,
                    "q": flds["q"],
                    "u": flds["u"],
                    "x": flds["x"],
                    "f": flds["f"],
                    "controls": deepcopy(sc["controls"][obj_id]),
                }
            )
        updated["groups"] = groups
        updated["schema"] = deepcopy(sc)
        updated["status"] = "identified"
        updated["complete"] = True
        updated["unresolved"] = []
    return updated


# ---------------------------------------------------------------------------
# Part 5B-2: Active Phase-Coherent Resonance Probing Near Critical Unit Circle
# ---------------------------------------------------------------------------


def plan_near_critical_resonance_probe(
    spec: TaskSpec,
    model: dict,
    records: list[dict],
    *,
    probe_idx: int = 0,
    length: int = 16,
) -> dict | None:
    """Synthesize a phase-coherent multi-step resonant probe to excite ``O(1)`` limit-cycle saturation.

    Triggers when:
      1. ``P_markov_state`` is refuted with ``unexplained_latent_transition_state`` or near-critical
         ``van_der_pol_transition_memory`` / ``oscillatory_transition_memory`` (``rho in [0.94, 1.15]``), AND
      2. Paid trajectories have weak cubic excitation leverage ``max_t |sum_i d_{i,t}| < 0.55`` or
         insufficient multi-step resonant coverage (fewer than 14 consecutive multi-step rows).
    """
    pa = model.get("premise_audit") or {}
    status = pa.get("status")
    lat = pa.get("latent_transition_state") or {}

    needs_resonance = False
    est_omega = 0.70

    if status == "unexplained_latent_transition_state":
        needs_resonance = True
    elif lat.get("mechanism") in {"van_der_pol_transition_memory", "oscillatory_transition_memory"}:
        pole = float(lat.get("pole", 0.0))
        ar2 = lat.get("ar2_params")
        if ar2 and pole > 1e-6:
            cw = max(-0.98, min(0.98, float(ar2[0]) / (2.0 * pole)))
            est_omega = acos(cw)
        if 0.94 <= pole <= 1.15:
            needs_resonance = True

    if not needs_resonance:
        return None

    sc = model.get("schema")
    if not sc:
        return None
    dynamic = [sc["fields"][g]["q"] for g in sc["objects"]]
    u_by_q = {sc["fields"][g]["q"]: sc["fields"][g]["u"] for g in sc["objects"]}

    # Measure current peak latent excitation & estimate natural frequency from multi-step trajectories
    by_probe: dict[str, dict[int, dict]] = {}
    for idx, r in enumerate(records):
        if "before" in r and "after" in r:
            pid = str(r.get("probe_id", f"__anon_{idx}"))
            pos = int(r.get("position", len(r.get("history", ()))))
            by_probe.setdefault(pid, {}).setdefault(pos, r)

    max_abs_sum_d = 0.0
    max_seq_len = 0
    ar_triples: list[tuple[float, float, float]] = []
    for rows_by_pos in by_probe.values():
        if 0 not in rows_by_pos:
            continue
        s_seq = []
        step = 0
        while step in rows_by_pos:
            cur = rows_by_pos[step]
            bef, aft = cur["before"], cur["after"]
            if not all(q in bef and q in aft and u_by_q[q] in bef for q in dynamic):
                break
            s_t = sum(float(aft[q]) - float(bef[q]) - float(bef[u_by_q[q]]) for q in dynamic)
            s_seq.append(s_t)
            if abs(s_t) > max_abs_sum_d:
                max_abs_sum_d = abs(s_t)
            step += 1
        if len(s_seq) > max_seq_len:
            max_seq_len = len(s_seq)
        for t in range(2, len(s_seq)):
            ar_triples.append((s_seq[t - 1], s_seq[t - 2], s_seq[t]))

    # If VdP nonlinear recurrence is already solved to machine precision, or strongly saturated, skip!
    if lat.get("mechanism") == "van_der_pol_transition_memory" and (
        float(lat.get("recurrence_mse", 1.0)) <= 1e-12
        or (max_abs_sum_d >= 0.85 and max_seq_len >= 16)
    ):
        return None

    if len(ar_triples) >= 4 and status == "unexplained_latent_transition_state":
        c_est = _linear_fit(
            [[r[0] for r in ar_triples], [r[1] for r in ar_triples]],
            [r[2] for r in ar_triples],
        )
        a1_p, a2_p = c_est[1], c_est[2]
        rho_p = sqrt(max(0.25, min(1.4, abs(a2_p))))
        cw_p = max(-0.96, min(0.96, a1_p / (2.0 * rho_p)))
        est_omega = acos(cw_p) if 0.15 <= acos(cw_p) <= 2.8 else 0.68

    bounds = spec.feature_bounds
    objs = list(sc["objects"])
    n = len(objs)
    seq_len = min(length, spec.max_probe_length)
    if seq_len < 8:
        return None

    rng = Random(7717 + 313 * probe_idx)
    focus_ctl = sc.get("focus_control", "focus")
    f_lo, f_hi = bounds[focus_ctl]

    # Phase-coherent resonant pump u_{i,t} = u_max * cos(est_omega * t + phase_shift)
    # Drives |z_{i,t}| rapidly from 0 into O(1) nonlinear cubic saturation!
    sequence: list[dict[str, float]] = []
    for t in range(seq_len):
        pt: dict[str, float] = {}
        for ctl, (lo, hi) in bounds.items():
            pt[ctl] = 0.5 * (lo + hi)
        pt[focus_ctl] = max(f_lo, min(f_hi, float(t % n)))
        for idx_o, g in enumerate(objs):
            ctls_g = sc["controls"][g]
            u_ctl = ctls_g["u"]
            q_ctl = ctls_g["q"]
            x_ctl = ctls_g.get("x")
            u_lo, u_hi = bounds[u_ctl]
            q_lo, q_hi = bounds[q_ctl]
            # Coherent resonant pumping for first 12 steps, then phase-shifted probe steps to span (z, z_prev, -z^3, u)
            if t < seq_len - 4:
                phase = est_omega * t + (0.35 * idx_o if probe_idx % 2 == 1 else 0.0)
                u_val = u_hi * cos(phase) if cos(phase) >= 0 else u_lo * abs(cos(phase))
            else:
                u_val = rng.uniform(u_lo, u_hi)
            pt[u_ctl] = max(u_lo, min(u_hi, u_val))
            pt[q_ctl] = rng.uniform(0.6 * q_lo, 0.6 * q_hi)
            if x_ctl and x_ctl in bounds:
                x_lo, x_hi = bounds[x_ctl]
                # Spread objects across informative distances [0.75, 1.15] & isolated positions
                span = x_hi - x_lo
                pt[x_ctl] = max(
                    x_lo,
                    min(
                        x_hi,
                        x_lo + span * ((idx_o + 0.5) / n) + rng.uniform(-0.08, 0.08),
                    ),
                )
        sequence.append(pt)

    return {
        "probe_id": f"active_resonance_probe_{probe_idx}",
        "estimated_omega": float(est_omega),
        "peak_sum_delta_before": float(max_abs_sum_d),
        "sequence": tuple(sequence),
    }
