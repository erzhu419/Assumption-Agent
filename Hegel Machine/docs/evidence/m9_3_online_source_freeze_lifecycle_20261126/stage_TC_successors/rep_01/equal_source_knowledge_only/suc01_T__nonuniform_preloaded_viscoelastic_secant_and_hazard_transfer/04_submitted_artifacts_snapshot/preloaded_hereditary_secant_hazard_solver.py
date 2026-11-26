"""Robust two-branch identification and first-passage prediction."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    n = len(train['lab_id'])
    x, ref, lab, strata = (np.concatenate((train[k], test[k]), axis=0) for k in
                           ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for group in np.unique(lab):
        at = lab == group
        panel = ref[at]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        x[at] = (x[at] - offset) / gain
    _, cell, count = np.unique(np.column_stack((lab, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    w = 1.0 / count[cell.ravel()]
    observed = np.isfinite(x)
    denom = w @ observed
    fill = (w @ np.where(observed, x, 0.0)) / np.where(denom > 0, denom, 1.0)
    x = np.where(observed, x, fill)
    x = np.column_stack((np.ones(len(x)), x))
    return x[:n], x[n:], w[:n] / w[:n].mean()


def _design(data, dt, t, x, tau):
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    weights = -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)
    memory = data['past_stress'] @ weights + data['initial_memory_states'] * np.exp(-dt.sum() / tau)
    gap = data['stress_hold_0'][:, None] - memory
    creep = -np.expm1(-t[:, None] / tau)
    curve = np.concatenate((gap[:, None, :] * creep, t[None, :, None] * x[:, None, :]), axis=2)
    return curve, np.column_stack((gap / tau, x))


def _fit(data, dt, t, x, w):
    n, k = data['strain_curves_obs'].shape
    y = np.r_[(data['strain_curves_obs'] - data['strain_0'][:, None]).ravel(), data['instant_rate_obs']]
    weight = np.r_[np.repeat(w, k), w]
    scale = np.r_[np.full(n * k, 0.025), np.full(n, 0.04)]
    target = y / scale
    root = np.sqrt(weight)
    p = 2 + x.shape[1]

    def matrix(tau):
        curve, rate = _design(data, dt, t, x, tau)
        return np.vstack((curve.reshape(n * k, p), rate)) / scale[:, None]

    def taus(q):
        fast = np.exp(q[0])
        return np.array([fast, np.exp(q[0] + q[1] * (np.log(9.) - q[0]))])

    def residual(q):
        r = matrix(taus(q)) @ q[2:] - target
        return root * np.copysign(np.sqrt(np.log1p(r * r)), r)

    def linear(a, weights):
        sw = np.sqrt(weights)
        return np.linalg.lstsq(a * sw[:, None], target * sw, rcond=None)[0]

    grid = np.geomspace(.20, 9., 9)
    starts = []
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            a = matrix(np.array([fast, slow]))
            b = linear(a, weight)
            for temp in (12., 6., 3., 1.5, 1.):
                b = linear(a, weight / (1 + ((a @ b - target) / temp) ** 2))
            r = a @ b - target
            q = np.r_[np.log(fast), np.clip(np.log(slow / fast) / np.log(9. / fast), 1e-6, 1.), b]
            starts.append((weight @ np.log1p(r * r), q))
    lo = np.r_[np.log(.20), 1e-6, np.full(p, -np.inf)]
    hi = np.r_[np.log(9.) - 1e-6, 1., np.full(p, np.inf)]
    best = min((least_squares(residual, q, bounds=(lo, hi), x_scale='jac', max_nfev=150)
                for _, q in sorted(starts, key=lambda z: z[0])[:3]), key=lambda z: z.cost)
    return taus(best.x), best.x[2:]


def _rate_update(train, test, dt, t, x, tau, coef, rate_test):
    """Condition the constant drift on the measured initial slope."""
    curve, rate = _design(train, dt, t, x, tau)
    # Late-time residuals divided by time estimate drift independently of
    # the rate sensor. A median across times suppresses curve glitches.
    late = slice(len(t) // 2, None)
    observed = train['strain_curves_obs'] - train['strain_0'][:, None]
    drift = np.median((observed[:, late] - curve[:, late] @ coef) / t[late], axis=1)
    rate_error = train['instant_rate_obs'] - rate @ coef - drift

    def mad_scale(v):
        return 1.4826 * np.median(np.abs(v - np.median(v)))

    missing_train = ~np.isfinite(train['coupon_features_raw'])
    missing_test = ~np.isfinite(test['coupon_features_raw'])
    rate_variance = mad_scale(rate_error)**2
    drift_variance = mad_scale(drift[~missing_train.any(axis=1)])**2
    # Missing slopes inherit the variation of observed coupons, not the
    # much smaller error typical of a complete calibrated coupon panel.
    coupon_variance = np.nanvar(np.where(missing_train, np.nan, x[:, 1:]), axis=0)
    prior_variance = drift_variance + missing_test @ (coef[3:]**2 * coupon_variance)
    gain = prior_variance / (prior_variance + rate_variance)
    innovation = test['instant_rate_obs'] - rate_test @ coef
    update = gain * innovation
    # Calibrate complete panels using training outcomes from the same lab.
    # Keep the pooled update for missing coupons and for a lab without outcomes.
    for lab in np.unique(train['lab_id']):
        calibration = (train['lab_id'] == lab) & ~missing_train.any(axis=1)
        apply = (test['lab_id'] == lab) & ~missing_test.any(axis=1)
        if not calibration.any() or not apply.any():
            continue
        center = np.median(drift[calibration])
        local_drift_var = mad_scale(drift[calibration])**2
        local_rate_var = mad_scale(rate_error[calibration])**2
        if local_drift_var + local_rate_var > 0:
            local_gain = local_drift_var / (local_drift_var + local_rate_var)
            update[apply] = center + local_gain * (innovation[apply] - center)
    return update


def _crossings(t, initial, curve, critical):
    result = np.full(len(initial), t[-1])
    reached = initial >= critical
    result[reached] = 0.0
    prev, left = initial, 0.0
    for right, current in zip(t, curve.T):
        hit = ~reached & (current >= critical)
        result[hit] = left + (right - left) * (critical[hit] - prev[hit]) / (current[hit] - prev[hit])
        reached |= hit
        prev, left = current, right
    return result


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x, xt, w = _features(train, test)
    tau, coef = _fit(train, dt, t, x, w)
    curve, rate = _design(test, dt, t, xt, tau)
    initial = test['strain_0']
    drift_update = _rate_update(train, test, dt, t, x, tau, coef, rate)
    strain = initial[:, None] + curve @ coef + drift_update[:, None] * t
    return {'estimated_taus': tau, 'predicted_strain': strain,
            'secant_compliance': (strain[:, -1] - initial) / np.maximum(np.abs(test['stress_hold_0']), .5),
            'hazard_crossing_times': _crossings(t, initial, strain, test['critical_strain'])}
