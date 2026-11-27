"""Calibrated two-branch hereditary model with crystallization-induced arrest."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = "/tmp/hegel_continual_archive/archive_calibration_history.json"


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate([train[k], test[k]]) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty_like(raw)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _calibrate(cal):
    gain, bias = np.ones(4), np.zeros(4)
    history = None
    for lab in range(4):
        x, y = (np.asarray(cal[key])[cal['lab_id'] == lab]
                for key in ('cmd_stress', 'true_stress'))
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            use = dx != 0
            if use.any():
                slope = np.median((y[j] - y[i])[use] / dx[use])
                r = y - slope*x
                clean = np.abs(r - np.median(r)) <= 3.0
        if len(x) < 2 or not clean.all():
            if history is None:
                with open(_ARCHIVE) as f:
                    history = json.load(f)
            if clean.sum() < 2:
                sweep = next(row for row in reversed(history) if lab in row['lab_id'])
                ref_lab, ref_x, ref_y = (np.asarray(sweep[key])
                                         for key in ('lab_id', 'cmd_stress', 'true_stress'))
                x, y = ref_x[ref_lab == lab], ref_y[ref_lab == lab]
                clean = np.ones(len(x), dtype=bool)
        gain[lab], bias[lab] = np.linalg.lstsq(
            np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]
    return gain, bias


def _design(split, x, gain, bias, dt, t, crit, banding):
    g, b = gain[split['lab_id']], bias[split['lab_id']]
    hist = g[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = g*split['stress_hold_cmd'] + b
    damage = np.maximum(np.abs(hist) - crit, 0)**2 @ dt
    arrest = (banding <= .50) & (damage > 0)
    act = np.where(arrest, np.tanh(damage), 0.)
    # Banding determines the flow basis; damage activates the network terms.
    terminal = np.log1p(t) if banding <= .50 else t
    flow = np.column_stack((np.ones(len(x)), x))
    age = np.cumsum(dt[::-1])[::-1] - dt
    curve = np.zeros((len(x), len(t), 9))
    rate = np.zeros((len(x), 9))
    curve[:, :, 2:7] = terminal[None, :, None]*flow[:, None, :]
    rate[:, 2:7] = flow

    def evaluate(log_tau):
        tau = np.exp(log_tau)
        kernel = np.exp(-age[:, None]/tau)*(-np.expm1(-dt[:, None]/tau))
        z = hist @ kernel
        gap = hold[:, None] - z
        rise = -np.expm1(-t[:, None]/tau)
        curve[:, :, :2] = gap[:, None, :]*rise
        rate[:, :2] = gap/tau
        curve[:, :, 7] = (act*hold)[:, None]*rise[:, 0]
        curve[:, :, 8] = (act*z[:, 0])[:, None]*terminal[None, :]
        rate[:, 7] = act*hold/tau[0]
        rate[:, 8] = act*z[:, 0]
        return curve, rate

    return evaluate, arrest


def _mad(r):
    return 1.4826*np.median(np.abs(r - np.median(r)))


def _fit(evaluate, curve_y, rate_y, dt, t):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    n = cm.sum()
    lo = np.log(min(dt.min(), t[t > 0].min())/20)
    hi = np.log(20*max(dt.sum(), t.max()))

    def matrix(log_tau):
        curve, rate = evaluate(log_tau)
        return np.concatenate((curve[cm], rate[rm]))

    w = np.r_[np.ones(n), np.full(rm.sum(), .5)]

    def project(log_tau):
        M = matrix(log_tau)*w[:, None]
        return np.linalg.lstsq(M, y*w, rcond=None)[0], M

    def profile(log_tau):
        beta, M = project(log_tau)
        return M @ beta - y*w

    starts = np.array([[.25, 2.], [.65, 4.], [1.4, 7.], [3., .4], [7., 1.4]])
    sols = [least_squares(profile, np.clip(np.log(start), lo, hi),
                          bounds=(lo, hi), max_nfev=60) for start in starts]
    best = min(sols, key=lambda r: r.cost)
    beta, _ = project(best.x)
    r = matrix(best.x) @ beta - y
    group_scale = [max(_mad(r[:n]), 1.e-6), max(_mad(r[n:]), 1.e-6)]
    scales = np.repeat(group_scale, [n, rm.sum()])

    def residual(p):
        return (matrix(p[:2]) @ p[2:] - y)/scales

    best = least_squares(residual, np.r_[best.x, beta], loss='cauchy', f_scale=2.5,
                         bounds=(np.r_[[lo]*2, [-np.inf]*9],
                                 np.r_[[hi]*2, [np.inf]*9]), max_nfev=100)
    return best.x[:2], best.x[2:]


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    xtr, xte = _align(train, test)
    args = gain, bias, dt, t, dataset['crit_stress'], dataset['shear_banding_index']
    build, _ = _design(train, xtr, *args)
    # Imputed coupons remain predictable, but do not inform the material fit
    # when complete coupons identify all nine linear coefficients.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    c0, r0 = build(np.log([.65, 4.]))
    # With no active network, its two coefficients are unidentifiable by design.
    columns = slice(None) if np.any(c0[:, :, 7:]) else slice(0, 7)
    design = np.vstack((c0[complete].reshape(-1, 9), r0[complete]))[:, columns]
    if not len(design) or np.linalg.matrix_rank(design) < design.shape[1]:
        complete[:] = True
    log_tau, beta = _fit(
        build, np.where(complete[:, None], train['strain_curves_obs'] - train['strain_0'][:, None], np.nan),
        np.where(complete, train['instant_rate_obs'], np.nan), dt, t)
    build, _ = _design(test, xte, *args)
    curve, rate = build(log_tau)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': beta[:7],
            'regime_mode_weights': beta[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ beta,
            'initial_rate_pred': rate @ beta}
