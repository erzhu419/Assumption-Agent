"""Robust identification of two variable-step hereditary creep branches."""

import numpy as np
from scipy.optimize import least_squares


def _panel(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = len(lab) / (len(count) * count[cell])
    for group in np.unique(lab):
        mask = lab == group
        panel = refs[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - offset) / gain
    valid = np.isfinite(x)
    mass = np.sum(weight[:, None] * valid, axis=0)
    weighted_sum = np.sum(weight[:, None] * np.where(valid, x, 0.0), axis=0)
    fill = np.divide(weighted_sum, mass, out=np.zeros_like(mass), where=mass > 0)
    return np.where(valid, x, fill), weight


def _design(tau, dt, tail, time, split, x):
    memory_weights = -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)
    stress_gap = split['stress_hold_0'][:, None] - split['past_stress'] @ memory_weights
    creep = -np.expm1(-time[:, None] / tau)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((stress_gap[:, None, :] * creep[None, :, :],
                            drift[:, None, :] * time[None, :, None]), axis=2)
    rate = np.column_stack((stress_gap / tau, drift))
    return curve, rate


def solve_hereditary_prony_source(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    x, weights = _panel(train, test)
    n = len(train['lab_id'])
    target = np.concatenate(((train['strain_curves_obs'] - train['strain_0'][:, None]).ravel(),
                             train['instant_rate_obs']))
    scale = np.r_[np.tile(0.012 + 0.008 * time, n), np.full(n, 0.020)]
    weight = np.r_[np.repeat(weights[:n], len(time)), weights[:n]]
    target = target / scale

    def taus(p):
        return np.array([0.20 + 7.8 * p[0], 0.20 + 7.8 * (p[0] + (1 - p[0]) * p[1])])

    def matrix(tau):
        curve, rate = _design(tau, dt, tail, time, train, x[:n])
        return np.vstack((curve.reshape(-1, 7), rate)) / scale[:, None]

    def weighted_fit(a, w):
        root = np.sqrt(w)
        return np.linalg.lstsq(a * root[:, None], target * root, rcond=None)[0]

    def residual(p):
        r = np.einsum('ij,j->i', matrix(taus(p)), p[2:]) - target
        # Square of this residual is weighted Cauchy loss. Applying the
        # cell weights outside the logarithm preserves the noise scales.
        return np.sqrt(weight) * np.copysign(np.sqrt(np.log1p(r * r)), r)

    starts = []
    grid = np.geomspace(0.22, 7.95, 11)
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            a = matrix(np.array([fast, slow]))
            coef = weighted_fit(a, weight)
            for anneal in (5., 3., 2., 1., 1.):
                r = np.einsum('ij,j->i', a, coef) - target
                coef = weighted_fit(a, weight / (1 + (r / anneal) ** 2))
            p = np.r_[(fast - 0.20) / 7.8, (slow - fast) / (8 - fast), coef]
            r = residual(p)
            starts.append((r @ r, p))

    lo = np.r_[0.0, 1e-7, np.full(7, -np.inf)]
    hi = np.r_[1 - 1e-7, 1.0, np.full(7, np.inf)]
    best = min((least_squares(residual, p, bounds=(lo, hi), x_scale='jac', max_nfev=150)
                for _, p in sorted(starts, key=lambda item: item[0])[:4]),
               key=lambda result: result.cost)
    tau, theta = taus(best.x), best.x[2:]
    curve, rate = _design(tau, dt, tail, time, test, x[n:])
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
