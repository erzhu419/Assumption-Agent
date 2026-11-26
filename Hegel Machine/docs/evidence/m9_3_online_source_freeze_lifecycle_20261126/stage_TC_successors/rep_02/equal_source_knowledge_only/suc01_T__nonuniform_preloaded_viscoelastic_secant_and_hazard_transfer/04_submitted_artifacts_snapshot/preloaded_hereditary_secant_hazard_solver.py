"""Two-branch hereditary transfer with a reference-panel drift model."""
import numpy as np
from scipy.optimize import least_squares


def _balanced(lab, stratum):
    _, ids, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    return len(lab) / (len(counts) * counts[ids])


def _panel(train, test):
    raw = np.concatenate([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    ref = np.concatenate([train['ref_coupons'], test['ref_coupons']])
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    for group in np.unique(lab):
        rows = lab == group
        panel = ref[rows]
        count = np.isfinite(panel).sum(axis=0)
        panel = panel[:, count >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        raw[rows] = (raw[rows] - offset) / gain
    w = _balanced(lab, stratum)
    observed = np.isfinite(raw)
    sums = (np.where(observed, raw, 0) * w[:, None]).sum(axis=0)
    mass = (observed * w[:, None]).sum(axis=0)
    fill = np.divide(sums, mass, out=np.zeros_like(sums), where=mass > 0)
    x = np.column_stack((np.ones(len(raw)), np.where(observed, raw, fill)))
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _design(data, dt, t, tau):
    # Exact expansion of the chronological piecewise-constant recurrence.
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)
    z = data['initial_memory_states'] * np.exp(-dt.sum() / tau) + data['past_stress'] @ kernel
    gap = data['stress_hold_0'][:, None] - z
    return gap[:, None, :] * -np.expm1(-t[:, None] / tau), gap / tau


def _fit(train, dt, times, x):
    n = len(x)
    curve = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate = train['instant_rate_obs']
    w = _balanced(train['lab_id'], train['stratum'])
    target = np.r_[(curve / .022).ravel(), rate / .023]
    root = np.sqrt(np.r_[np.repeat(w, len(times)), w])
    drift = (x[:, None, :] * (times / .022)[None, :, None]).reshape(n * len(times), -1)
    drift = np.vstack((drift, x / .023))

    def design(tau):
        a, b = _design(train, dt, times, tau)
        return np.column_stack((np.vstack(((a / .022).reshape(-1, 2), b / .023)), drift))

    lo, hi = np.log([.20, 9.])

    def decode(p):
        return np.exp([p[0], p[0] + (hi - p[0]) * p[1]])

    def residual(p):
        r = np.einsum('ij,j->i', design(decode(p)), p[2:]) - target
        return root * np.copysign(np.sqrt(np.log1p(r * r)), r)

    starts = []
    grid = np.geomspace(.205, 8.95, 10)
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            lf, ls = np.log([fast, slow])
            p = [lf, (ls - lf) / (hi - lf)]
            a = design(np.array([fast, slow]))
            v = root
            for temp in (np.inf, 8., 4., 2., 1., 1.):
                coef = np.linalg.lstsq(a * v[:, None], target * v, rcond=None)[0]
                r = np.einsum('ij,j->i', a, coef) - target
                v = root / np.hypot(1., r / temp)
            p = np.r_[p, coef]
            r = residual(p)
            starts.append((r @ r, p))
    bounds = (np.r_[lo, 1e-6, np.full(len(coef), -np.inf)],
              np.r_[hi - 1e-6, 1., np.full(len(coef), np.inf)])
    result = min((least_squares(residual, p, bounds=bounds, x_scale='jac', max_nfev=130)
                  for _, p in sorted(starts, key=lambda pair: pair[0])[:4]), key=lambda fit: fit.cost)
    return decode(result.x), result.x[2:]


def _rate_weight(train, dt, times, x, taus, coef):
    """Calibrate the drift blend from late curves, without using held-out outcomes."""
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    if not complete.any():
        return 1.
    curve, rate = _design(train, dt, times, taus)
    increment = train['strain_curves_obs'] - train['strain_0'][:, None]
    late = times >= np.median(times)
    # A median over the late samples estimates drift without differentiating noisy curves.
    drift_curve = np.median((increment[:, late] - curve[:, late] @ coef[:2]) / times[late], axis=1)
    drift_coupon = x @ coef[2:]
    delta = (train['instant_rate_obs'] - rate @ coef[:2] - drift_coupon)[complete]
    target = (drift_curve - drift_coupon)[complete]
    weight = _balanced(train['lab_id'][complete], train['stratum'][complete])
    blend = .5
    for _ in range(8):
        v = weight / np.hypot(1., (blend * delta - target) / .012)
        denominator = np.dot(v, delta * delta)
        if denominator == 0:
            break
        blend = np.clip(np.dot(v, delta * target) / denominator, 0., 1.)
    return blend


def _crossings(initial, curves, times, critical):
    answer = np.full(len(initial), times[-1], dtype=float)
    reached = initial >= critical
    answer[reached] = 0.
    prev_value, prev_time = initial, 0.
    for j, time in enumerate(times):
        value = curves[:, j]
        hit = ~reached & (value >= critical)
        fraction = (critical[hit] - prev_value[hit]) / (value[hit] - prev_value[hit])
        answer[hit] = prev_time + (time - prev_time) * fraction
        reached |= hit
        prev_value, prev_time = value, time
    return answer


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    x, xx = _panel(train, test)
    taus, coef = _fit(train, dt, times, x)
    a, b = _design(test, dt, times, taus)
    drift_coupon = xx @ coef[2:]
    drift_rate = test['instant_rate_obs'] - b @ coef[:2]
    complete = np.isfinite(test['coupon_features_raw']).all(axis=1)
    blend = _rate_weight(train, dt, times, x, taus, coef)
    # A measured rate is preferable to a population fill for a missing coupon.
    drift = drift_rate + np.where(complete, 1. - blend, 0.) * (drift_coupon - drift_rate)
    strain = test['strain_0'][:, None] + a @ coef[:2] + drift[:, None] * times
    return {'estimated_taus': taus, 'predicted_strain': strain,
            'secant_compliance': (strain[:, -1] - test['strain_0']) / np.maximum(np.abs(test['stress_hold_0']), .5),
            'hazard_crossing_times': _crossings(test['strain_0'], strain, times, test['critical_strain'])}
