"""Fit two hereditary branches and two yield-activated strain terms."""

import numpy as np
from scipy.optimize import least_squares


def _cell_weights(lab, stratum):
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    return len(lab) / (len(counts) * counts[cell.ravel()])


def _align(train, test):
    x = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    ref = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    lab = np.concatenate((train['lab_id'], test['lab_id']))
    strata = np.concatenate((train['stratum'], test['stratum']))
    for group in np.unique(lab):
        rows = lab == group
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        x[rows] = (x[rows] - center) / gain
    w = _cell_weights(lab, strata)[:, None]
    valid = np.isfinite(x)
    total = np.sum(w * np.where(valid, x, 0.0), axis=0)
    mass = np.sum(w * valid, axis=0)
    fill = np.divide(total, mass, out=np.zeros_like(total), where=mass > 0)
    return np.where(valid, x, fill)


def _inputs(split, aligned, dt, threshold):
    history = np.asarray(split['past_stress'])
    hold = np.asarray(split['stress_hold_0'])
    creep = np.sign(hold) * np.maximum(np.abs(hold) - threshold, 0.0)**2
    work = (np.maximum(np.abs(history) - threshold, 0.0)**2) @ dt
    drift = np.column_stack((np.ones(len(hold)), aligned, creep))
    return history, hold, drift, work * hold


def _design(inputs, dt, times, tau):
    history, hold, drift, damage = inputs
    # Piecewise-constant history, oldest interval first, zero initial memory.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = -np.expm1(-dt[:, None] / tau) * np.exp(-remaining[:, None] / tau)
    gap = hold[:, None] - history @ kernel
    rise = -np.expm1(-times[:, None] / tau)
    curve = np.concatenate((gap[:, None, :] * rise[None, :, :],
                            drift[:, None, :] * times[None, :, None],
                            damage[:, None, None] * rise[None, :, :1]), axis=2)
    rate = np.column_stack((gap / tau, drift, damage / tau[0]))
    return curve, rate


def _fit(inputs, dt, times, increments, rates, weights, plastic=True):
    n, k = increments.shape
    target = np.r_[increments.ravel(), rates]
    root = np.sqrt(np.r_[np.repeat(weights, k), weights])
    cols = slice(None) if plastic else slice(None, -2)
    count = 9 if plastic else 7

    def decode(q):
        fast = 0.20 * np.exp(np.log(40.0) * q[0])
        return np.array((fast, fast * np.exp(np.log(8.0 / fast) * q[1])))

    def matrix(q):
        curve, rate = _design(inputs, dt, times, decode(q))
        return np.vstack((curve.reshape(n * k, -1), rate))[:, cols]

    def linear(a, w):
        return np.linalg.lstsq(a * w[:, None], target * w, rcond=None)[0]

    def raw(a, coef):
        return np.einsum('ij,j->i', a, coef) - target

    def build_residual(scale):
        def residual(p):
            r = raw(matrix(p[:2]), p[2:]) / scale
            return root * np.copysign(np.sqrt(np.log1p(r * r)), r)
        return residual

    # First fit at conservative noise scales, then estimate the clean core
    # in each observation channel from its median absolute residual.
    scale = np.r_[np.full(n * k, 0.055), np.full(n, 0.035)]
    residual = build_residual(scale)
    starts = []
    grid = np.geomspace(0.20, 8.0, 9)
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            q = np.array((np.log(fast / .20) / np.log(40.0),
                          np.log(slow / fast) / np.log(8.0 / fast)))
            q = np.clip(q, (0.0, 1e-5), (1.0 - 1e-5, 1.0))
            a = matrix(q)
            w = root / scale
            coef = linear(a, w)
            for temperature in (5.0, 2.5, 1.0, 1.0):
                r = raw(a, coef) / (scale * temperature)
                coef = linear(a, w / np.hypot(1.0, r))
            p = np.r_[q, coef]
            r = residual(p)
            starts.append((r @ r, p))

    bounds = (np.r_[0.0, 1e-5, np.full(count, -np.inf)],
              np.r_[1.0 - 1e-5, 1.0, np.full(count, np.inf)])

    def refine(f, p):
        return least_squares(f, p, bounds=bounds, x_scale='jac',
                             tr_solver='lsmr', max_nfev=120, ftol=1e-8)

    best = min((refine(residual, p) for _, p in sorted(starts, key=lambda v: v[0])[:3]),
               key=lambda fit: fit.cost)
    errors = raw(matrix(best.x[:2]), best.x[2:])
    scale = np.r_[np.full(n * k, max(0.008, 1.4826 * np.median(np.abs(errors[:n * k])))),
                  np.full(n, max(0.008, 1.4826 * np.median(np.abs(errors[n * k:]))))]
    best = refine(build_residual(scale), best.x)
    return decode(best.x[:2]), best.x[2:]


def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    """Return ordered relaxation times, plastic weights, and clean hold curves."""
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    threshold = float(dataset['yield_stress'])
    n = len(train['lab_id'])
    aligned = _align(train, test)
    weights = _cell_weights(train['lab_id'], train['stratum'])
    inputs = _inputs(train, aligned[:n], dt, threshold)
    increments = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, coef = _fit(inputs, dt, times, increments, train['instant_rate_obs'], weights)
    curve, rate = _design(_inputs(test, aligned[n:], dt, threshold), dt, times, tau)
    prediction = test['strain_0'][:, None] + curve @ coef
    # Missing coupon features affect only the steady drift. Use the measured
    # onset rate to recover that drift without changing the memory or plastic terms.
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    drift_error = test['instant_rate_obs'][missing] - rate[missing] @ coef
    prediction[missing] += drift_error[:, None] * times
    return {'estimated_taus': tau, 'viscoplastic_weights': coef[-2:].copy(),
            'predicted_strain': prediction}
