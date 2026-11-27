"""Calibrated two-branch hereditary creep for sub-yield coupons."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _load_cell_fit(checks):
    gain, bias = np.ones(4), np.zeros(4)
    for lab in range(4):
        rows = checks['lab_id'] == lab
        x, y = checks['cmd_stress'][rows], checks['true_stress'][rows]
        i, j = np.triu_indices(x.size, 1)
        use = np.abs(x[j] - x[i]) > 1e-8
        if not np.any(use):
            if x.size:
                bias[lab] = np.median(y - x)
            continue
        slopes = (y[j[use]] - y[i[use]]) / (x[j[use]] - x[i[use]])
        g = np.median(slopes)
        b = np.median(y - g * x)
        r = y - g * x - b
        scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-4)
        keep = np.abs(r) < 5 * scale
        if np.count_nonzero(keep) >= 2 and np.ptp(x[keep]) > 1e-8:
            g, b = np.linalg.lstsq(np.c_[x[keep], np.ones(keep.sum())], y[keep], rcond=None)[0]
        gain[lab], bias[lab] = g, b
    return gain, bias


def _align(train, test):
    lab, strata, features, ref = [np.concatenate((train[k], test[k])) for k in
                                  ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons')]
    X = features.astype(float, copy=True)
    cell_means = []
    for l in np.unique(lab):
        rows = lab == l
        panel = ref[rows]
        usable = np.any(np.isfinite(panel), axis=0)
        center = np.median(np.nanmean(panel[:, usable], axis=0))
        spread = np.median(np.nanstd(panel[:, usable], axis=0))
        X[rows] = (X[rows] - center) / spread
        for s in np.unique(strata[rows]):
            cell = rows & (strata == s)
            values = X[cell]
            observed = np.isfinite(values)
            count = observed.sum(axis=0)
            total = np.where(observed, values, 0.).sum(axis=0)
            cell_means.append(np.divide(total, count, out=np.full(X.shape[1], np.nan), where=count > 0))
    means = np.asarray(cell_means)
    available = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), available, out=np.zeros(X.shape[1]), where=available > 0)
    return np.where(np.isfinite(X), X, fill)


def _design(taus, dt, times, past, hold, X):
    # End-of-history convolution, including relaxation after each segment.
    after = np.cumsum(dt[::-1])[::-1] - dt
    memory = -np.expm1(-dt[:, None] / taus) * np.exp(-after[:, None] / taus)
    delta = hold[:, None] - np.einsum('ns,sm->nm', past, memory)
    creep = delta[:, None, :] * -np.expm1(-times[None, :, None] / taus)
    drift = np.c_[np.ones(len(X)), X]
    curves = np.concatenate((creep, times[None, :, None] * drift[:, None, :]), axis=2)
    return curves, np.concatenate((delta / taus, drift), axis=1)


def _mad(r):
    return max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-7)


def _fit(dt, times, past, hold, X, y, rate):
    n = len(X)
    observed_curve = np.isfinite(y) & (times[None, :] > 0)
    observed_rate = np.isfinite(rate)
    target = np.r_[y[observed_curve], rate[observed_rate]]
    k = observed_curve.sum()

    def design(taus):
        A, B = _design(taus, dt, times, past, hold, X)
        return np.concatenate((A[observed_curve], B[observed_rate]))

    def estimate(A, w):
        return np.linalg.lstsq(A * w[:, None], target * w, rcond=None)[0]

    def scaled(r):
        return np.r_[np.full(k, _mad(r[:k])), np.full(len(r)-k, _mad(r[k:]))]

    # Both branches are resolved globally on a small initial time-scale grid.
    grid = np.geomspace(.2, 8., 9)
    candidates = []
    for i, a in enumerate(grid[:-1]):
        for b in grid[i+1:]:
            A = design(np.array([a, b]))
            w = np.ones(len(target))
            for _ in range(3):
                coef = estimate(A, w)
                r = np.einsum('ij,j->i', A, coef) - target
                s = scaled(r)
                w = 1 / s / np.sqrt(1 + (r / (2.4*s))**2)
            candidates.append((len(r)*np.mean(np.log(s)), a, b, coef, s))
    candidates.sort(key=lambda p: p[0])
    lo, hi = np.log([.2, 8.])

    def taus(p):
        return np.exp([p[0], p[0] + (hi - p[0]) * p[1]])

    def refine(p, s):
        s = 2.4 * s
        def residual(p):
            return (np.einsum('ij,j->i', design(taus(p)), p[2:]) - target) / s
        return least_squares(residual, p, bounds=([lo, 1e-6]+[-np.inf]*7,
                             [hi-1e-6, 1.]+[np.inf]*7), loss='cauchy',
                             x_scale='jac', max_nfev=110, ftol=1e-8)

    _, _, _, _, common_scale = candidates[0]
    fits = [refine(np.r_[np.log(a), (np.log(b)-np.log(a))/(hi-np.log(a)), c], common_scale)
            for _, a, b, c, _ in candidates[:4]]
    p = min(fits, key=lambda f: f.cost).x
    for _ in range(2):
        s = scaled(np.einsum('ij,j->i', design(taus(p)), p[2:]) - target)
        p = refine(p, s).x
    return taus(p), p[2:]


def solve_alpha_m96_w03_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    X = _align(train, test)
    gain, bias = _load_cell_fit(dataset['calibration_load_cells'])
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])

    def calibrated(split):
        lab = split['lab_id']
        return (split['past_stress_cmd'] * gain[lab, None] + bias[lab, None],
                split['stress_hold_cmd'] * gain[lab] + bias[lab])

    n = len(train['lab_id'])
    ts, theta = _fit(dt, times, *calibrated(train), X[:n],
                     train['strain_curves_obs'] - train['strain_0'][:, None],
                     train['instant_rate_obs'])
    A, B = _design(ts, dt, times, *calibrated(test), X[n:])
    return {'estimated_taus': ts, 'hereditary_weights': theta,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + np.einsum('nkj,j->nk', A, theta),
            'initial_rate_pred': B @ theta}
