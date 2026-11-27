"""Fit a calibrated two-branch creep model and predict deployment coupons."""
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares


def _calibration(cal):
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    fit = np.linalg.lstsq(np.column_stack((np.ones(len(loads)), loads)), raw, rcond=None)[0]
    # Only the sum of the four corrected channels enters the constitutive law.
    return fit[0], np.linalg.solve(fit[1:], np.ones(loads.shape[1]))


def _features(train, test):
    lab = np.concatenate((train['lab_id'], test['lab_id']))
    stratum = np.concatenate((train['stratum'], test['stratum']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for group in np.unique(lab):
        rows = lab == group
        controls = ref[rows]
        center = np.median(np.mean(controls, axis=0))
        scale = np.median(np.std(controls, axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell].reshape(-1, 1)
    observed = ~np.isnan(x)
    filled = np.where(observed, x, 0.0)
    total = (weights * observed).sum(axis=0)
    means = np.divide((weights * filled).sum(axis=0), total,
                      out=np.zeros(x.shape[1]), where=total > 0)
    return np.where(observed, x, means)


def _prepare(dataset):
    tr, te = dataset['train'], dataset['test']
    x = _features(tr, te)
    offset, total = _calibration(dataset['calibration_load_cells'])
    n = len(tr['lab_id'])
    def coupon(data, features):
        return ((data['past_channel_raw'] - offset) @ total,
                (data['hold_channel_raw'] - offset) @ total,
                np.column_stack((np.ones(len(features)), features)))
    return coupon(tr, x[:n]), coupon(te, x[n:])


def _design(coupon, dt, times, taus):
    past, hold, drift = coupon
    z = np.zeros((len(hold), 2))
    for step, duration in enumerate(dt):
        w = -np.expm1(-duration / taus)
        z += w * (past[:, step, None] - z)
    gap = hold[:, None] - z
    a = np.empty((len(hold), len(times) + 1, 7))
    a[:, :-1, :2] = gap[:, None, :] * (-np.expm1(-times[:, None] / taus))
    a[:, -1, :2] = gap / taus
    a[:, :-1, 2:] = drift[:, None, :] * times[None, :, None]
    a[:, -1, 2:] = drift
    return a


def _fit(coupon, dt, times, y, loss='cauchy', tuning=2.5):
    n, k = y.shape
    lo, hi = np.log([0.05000001, 24.999999])
    bounds = (np.r_[lo, lo, np.full(7, -np.inf)],
              np.r_[hi, hi, np.full(7, np.inf)])
    def linear_seed(taus):
        a = _design(coupon, dt, times, taus).reshape(-1, 7)
        coef = np.linalg.lstsq(a, y.ravel(), rcond=None)[0]
        return np.sum((a @ coef - y.ravel())**2), np.r_[np.log(taus), coef]
    seeds = sorted((linear_seed(np.array(pair)) for pair in combinations(
        (0.075, 0.15, 0.3, 0.55, 0.9, 1.6, 3., 6., 12., 22.), 2)), key=lambda p: p[0])
    def residual(p):
        return _design(coupon, dt, times, np.exp(p[:2])) @ p[2:] - y
    def optimize(seed, scale, kind):
        return least_squares(lambda p: (residual(p) / scale).ravel(), seed,
                             method='trf', bounds=bounds, loss=kind, x_scale='jac',
                             max_nfev=130, ftol=1e-9)
    seed = seeds[0][1]
    def scales(p):
        r = residual(p)
        curve, rate = r[:, :-1].ravel(), r[:, -1]
        return np.maximum([1.4826 * np.median(abs(v - np.median(v)))
                           for v in (curve, rate)], 1e-8)
    def weights(s):
        return np.r_[np.repeat(s[0], k - 1), s[1]]
    if loss == 'linear':
        p = optimize(seed, np.ones(k), 'linear').x
    else:
        p = optimize(seed, 1.5 * weights(scales(seed)), 'soft_l1').x
        p = optimize(p, tuning * weights(scales(p)), loss).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_batch(dataset: dict) -> dict:
    """Predict deployment strain and its initial derivative from benchtop data."""
    train, test = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tr = dataset['train']
    y = np.column_stack((tr['strain_curves_obs'] - tr['strain_0'][:, None],
                         tr['instant_rate_obs']))
    taus, theta = _fit(train, dt, times, y)
    pred = _design(test, dt, times, taus) @ theta
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': dataset['test']['strain_0'][:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
