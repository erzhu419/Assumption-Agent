"""Robust identification of two sub-yield hereditary creep branches."""
import numpy as np
from scipy.optimize import least_squares


def _calibration(checks):
    labs = np.asarray(checks['lab_id'])
    gain, bias = np.full(4, np.nan), np.full(4, np.nan)
    for lab in range(4):
        x = np.asarray(checks['cmd_stress'])[labs == lab]
        y = np.asarray(checks['true_stress'])[labs == lab]
        if len(x) < 2 or np.ptp(x) < 1e-8:
            continue
        i, j = np.triu_indices(len(x), 1)
        dx = x[j] - x[i]
        far = np.abs(dx) >= 0.1 * np.ptp(x)
        slope = np.median((y[j[far]] - y[i[far]]) / dx[far])
        r = y - slope * x
        center = np.median(r)
        spread = 1.4826 * np.median(np.abs(r - center))
        good = np.abs(r - center) <= max(4 * spread, 1e-7)
        if good.sum() >= 2 and np.ptp(x[good]) > 1e-8:
            slope, center = np.linalg.lstsq(np.c_[x[good], np.ones(good.sum())], y[good], rcond=None)[0]
        gain[lab], bias[lab] = slope, center
    for lab in np.flatnonzero(~np.isfinite(gain)):
        peer = lab ^ 1
        gain[lab] = gain[peer] if np.isfinite(gain[peer]) else 1.
        bias[lab] = bias[peer] if np.isfinite(bias[peer]) else 0.
        if np.any(labs == lab):
            bias[lab] = np.median(np.asarray(checks['true_stress'])[labs == lab] - gain[lab] * np.asarray(checks['cmd_stress'])[labs == lab])
    return gain, bias


def _align(train, test):
    labs, strata, raw, ref = (np.concatenate([train[k], test[k]]) for k in
                               ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    x = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        m = labs == lab
        loc = np.nanmedian(np.nanmean(ref[m], axis=0))
        scale = np.nanmedian(np.nanstd(ref[m], axis=0))
        x[m] = (raw[m] - loc) / scale
    cells = [x[(labs == lab) & (strata == st)] for lab, st in np.unique(np.c_[labs, strata], axis=0)]
    fill = np.nanmean([np.nanmean(v, axis=0) for v in cells], axis=0)
    return np.where(np.isnan(x), fill, x)


def _inputs(split, x, gain, bias):
    lab = np.asarray(split['lab_id'])
    past = gain[lab, None] * split['past_stress_cmd'] + bias[lab, None]
    hold = gain[lab] * split['stress_hold_cmd'] + bias[lab]
    return past, hold, np.c_[np.ones(len(lab)), x]


def _basis(tau, dt, t, past, hold):
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for p, step in zip(past.T, dt):
        e = np.exp(-step / tau)
        dz = e * dz + (z - p[:, None]) * (e * step / tau**2)
        z = e * z - np.expm1(-step / tau) * p[:, None]
    a = hold[:, None] - z
    tq = t[:, None] / tau
    b = -np.expm1(-tq)
    db = -np.exp(-tq) * tq / tau
    return a[:, None, :] * b, -dz[:, None, :] * b + a[:, None, :] * db, a / tau, -dz / tau - a / tau**2


def _fit(dt, t, args, y, rate):
    past, hold, x = args
    n, k = y.shape
    tail = np.concatenate([(x[:, None, :] * t[None, :, None]).reshape(-1, 5), x])
    target = np.r_[y.ravel(), rate]

    def design(tau):
        b, db, r, dr = _basis(tau, dt, t, past, hold)
        return np.c_[np.concatenate((b.reshape(-1, 2), r)), tail], np.concatenate((db.reshape(-1, 2), dr))

    def solve(start, scales, loss):
        def residual(p):
            d, _ = design(np.exp(p[:2]))
            return (d @ p[2:] - target) / scales

        def jacobian(p):
            tau = np.exp(p[:2])
            d, dd = design(tau)
            return np.c_[dd * (tau * p[2:4]), d] / scales[:, None]

        return least_squares(residual, start, jac=jacobian, loss=loss, bounds=(
            np.r_[np.log([.2, .2]), np.full(7, -np.inf)],
            np.r_[np.log([8., 8.]), np.full(7, np.inf)]), x_scale='jac', max_nfev=100)

    def scale_from(fit):
        residual = design(np.exp(fit.x[:2]))[0] @ fit.x[2:] - target
        curves, onset = residual[:n*k], residual[n*k:]
        def mad(v):
            return max(1e-7, 1.4826 * np.median(np.abs(v - np.median(v))))
        return np.r_[np.full(n*k, 2.5 * mad(curves)), np.full(n, 2.5 * mad(onset))]

    scales = np.full(len(target), .025)
    # Profile the linear weights on a coarse, logarithmic time-constant grid.
    # A robust profile keeps a large strain glitch out of the nonlinear seed.
    grid = np.geomspace(.201, 7.99, 8)
    seeds = []
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i+1:]:
            tau = np.array([fast, slow])
            d, _ = design(tau)
            w = np.linalg.lstsq(d, target, rcond=None)[0]
            for _ in range(16):
                root = np.hypot(1., (d @ w - target) / scales)**(-.5)
                w = np.linalg.lstsq(d * root[:, None], target * root, rcond=None)[0]
            r = (d @ w - target) / scales
            cost = np.sum(np.hypot(1., r) - 1.)
            seeds.append((cost, np.r_[np.log(tau), w]))
    seeds.sort(key=lambda item: item[0])
    fits = [solve(p, scales, 'soft_l1') for _, p in seeds[:3]]
    seed = min(fits, key=lambda f: f.cost)
    fit = solve(seed.x, scale_from(seed), 'cauchy')
    fit = solve(fit.x, scale_from(fit), 'cauchy')
    tau = np.exp(fit.x[:2])
    order = np.argsort(tau)
    return tau[order], np.r_[fit.x[2:4][order], fit.x[4:]]


def solve_alpha_m96_w02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    gain, bias = _calibration(dataset['calibration_load_cells'])
    x = _align(train, test)
    n = len(train['lab_id'])
    args = _inputs(train, x[:n], gain, bias)
    tau, theta = _fit(dt, t, args, train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    past, hold, xte = _inputs(test, x[n:], gain, bias)
    b, _, r, _ = _basis(tau, dt, t, past, hold)
    drift = xte @ theta[2:]
    return {'estimated_taus': tau, 'hereditary_weights': theta, 'viscoplastic_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + b @ theta[:2] + drift[:, None] * t,
            'initial_rate_pred': r @ theta[:2] + drift}
