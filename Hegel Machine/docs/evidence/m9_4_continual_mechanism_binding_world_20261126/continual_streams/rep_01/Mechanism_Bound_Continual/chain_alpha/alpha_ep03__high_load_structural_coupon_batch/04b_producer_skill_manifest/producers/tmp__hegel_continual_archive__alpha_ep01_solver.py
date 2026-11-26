"""Calibrated two-branch hereditary creep with preloaded memory."""
import numpy as np
from scipy.optimize import least_squares


def _aligned_panels(dataset):
    train, test = dataset['train'], dataset['test']
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for lab in np.unique(labs):
        mask = labs == lab
        r = refs[mask]
        x[mask] = (x[mask] - np.nanmedian(np.nanmean(r, axis=0))) / np.nanmedian(np.nanstd(r, axis=0, ddof=0))
    complete = np.isfinite(x).all(axis=1)
    # Each lab/stratum cell contributes one mean, regardless of its enrollment.
    _, cells = np.unique(np.column_stack([labs, strata]), axis=0, return_inverse=True)
    means = np.stack([np.nanmean(x[cells == c], axis=0) for c in np.unique(cells)])
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    return x, complete


def _physical_stresses(dataset, panel):
    cal = dataset['calibration_load_cells']
    labels = np.asarray(panel['lab_id'])
    past = np.empty_like(panel['past_stress_cmd'], dtype=float)
    hold = np.empty_like(panel['stress_hold_cmd'], dtype=float)
    for lab in np.unique(labels):
        fit = np.asarray(cal['lab_id']) == lab
        cmd = np.asarray(cal['cmd_stress'])[fit]
        gain, bias = np.linalg.lstsq(np.column_stack([cmd, np.ones_like(cmd)]), np.asarray(cal['true_stress'])[fit], rcond=None)[0]
        rows = labels == lab
        past[rows] = gain * np.asarray(panel['past_stress_cmd'])[rows] + bias
        hold[rows] = gain * np.asarray(panel['stress_hold_cmd'])[rows] + bias
    return past, hold


class _Design:
    def __init__(self, past, hold, dt, times, x, initial):
        self.initial = np.asarray(initial)
        self.past = past
        self.hold = hold
        self.dt = np.asarray(dt)
        self.tail = np.cumsum(self.dt[::-1])[::-1] - self.dt
        self.t = np.asarray(times)
        self.drift = np.column_stack([np.ones(len(x)), x])[:, None, :] * np.r_[self.t, 1.0][None, :, None]

    def matrix(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u, v, h = self.dt[:, None] / tau, self.tail[:, None] / tau, self.t[:, None] / tau
        tail_decay = np.exp(-v)
        weights = tail_decay * -np.expm1(-u)
        preload = self.initial * np.exp(-self.dt.sum() / tau)
        gap = self.hold[:, None] - self.past @ weights - preload
        time_basis = np.vstack([-np.expm1(-h), 1 / tau])
        branch = gap[:, None, :] * time_basis[None, :, :]
        a = np.concatenate([branch, self.drift], axis=2)
        if not derivative:
            return a
        dgap = -self.past @ (v * weights - u * np.exp(-u) * tail_decay) - preload * (self.dt.sum() / tau)
        dtime = np.vstack([-h * np.exp(-h), -1 / tau])
        return a, dgap[:, None, :] * time_basis[None, :, :] + gap[:, None, :] * dtime[None, :, :]


def _mad(r):
    return np.maximum(1e-8, 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0))


def _fit(design, y, fit_rows):
    y = y[fit_rows]
    log_bounds = np.log([0.20, 8.0])
    lo, hi = np.r_[[log_bounds[0]] * 2, [-np.inf] * 7], np.r_[[log_bounds[1]] * 2, [np.inf] * 7]
    grid = np.geomspace(0.20, 8.0, 8)
    starts = []
    for i, t1 in enumerate(grid):
        for t2 in grid[i + 1:]:
            l = np.log([t1, t2])
            a = design.matrix(l)[fit_rows]
            coef = np.linalg.lstsq(a.reshape(-1, 7), y.ravel(), rcond=None)[0]
            r = a @ coef - y
            starts.append((np.sum(r * r), np.r_[l, coef], r))
    starts.sort(key=lambda s: s[0])

    def optimize(p, scale, loss, f_scale):
        def residual(p):
            a = design.matrix(p[:2])[fit_rows]
            return ((a @ p[2:] - y) / scale).ravel()

        def jac(p):
            a, da = design.matrix(p[:2], derivative=True)
            full = np.concatenate([da[fit_rows] * p[2:4], a[fit_rows]], axis=2)
            return (full / scale[..., None]).reshape(-1, 9)

        return least_squares(residual, p, jac=jac, bounds=(lo, hi), loss=loss, f_scale=f_scale, max_nfev=180, x_scale='jac')

    def noise(r):
        return np.r_[np.full(r.shape[1] - 1, _mad(r[:, :-1].ravel())), _mad(r[:, -1])]

    warm_scale = noise(starts[0][2])
    pilots = [optimize(s[1], warm_scale, 'soft_l1', 1.5) for s in starts[:3]]
    best = min(pilots, key=lambda f: f.cost)
    raw = (design.matrix(best.x[:2])[fit_rows] @ best.x[2:]) - y
    scale = noise(raw)
    result = min((optimize(p.x, scale, 'cauchy', 2.0) for p in pilots), key=lambda f: f.cost)
    p = result.x
    return np.exp(p[:2]), p[2:]


def _prepare(dataset):
    x, complete = _aligned_panels(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    dtrain = _Design(*_physical_stresses(dataset, train), dt, times, x[:n], train['initial_memory_states'])
    dtest = _Design(*_physical_stresses(dataset, test), dt, times, x[n:], test['initial_memory_states'])
    y = np.column_stack([train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None], train['instant_rate_obs']])
    return dtrain, dtest, y, complete[:n]


def _rate_weights(residual, complete, times):
    # The initial rate measures the same drift as the slope across hold times.
    # Estimate its reliability from a late-time slope resistant to isolated spikes.
    late = slice(len(times) // 2, None)
    slope = np.median(residual[:, :-1][:, late] / times[late], axis=1)
    noise = _mad((residual[:, -1] - slope)[complete])
    weights = []
    for group in (complete, ~complete):
        # Estimate the variance attributable to drift in the rate innovation.
        spread = _mad(residual[group, -1]) if group.sum() >= 5 else noise
        weights.append(max(0.0, 1.0 - (noise / spread)**2))
    return weights


def _update_drift(pred, observed_rate, complete, times, weights):
    delta = np.asarray(observed_rate) - pred[:, -1]
    correction = np.where(complete, weights[0], weights[1]) * np.where(np.isfinite(delta), delta, 0.0)
    return pred + correction[:, None] * np.r_[times, 1.0]


def _crossing_times(strain_0, curves, critical_strain, times):
    values = np.column_stack([strain_0, curves])
    t = np.r_[0.0, times]
    hit = values >= np.asarray(critical_strain)[:, None]
    first = np.argmax(hit, axis=1)
    result = np.full(len(values), times[-1], dtype=float)
    result[hit[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    lo, hi = values[rows, j - 1], values[rows, j]
    fraction = (np.asarray(critical_strain)[rows] - lo) / (hi - lo)
    result[rows] = t[j - 1] + fraction * (t[j] - t[j - 1])
    return result


def solve_alpha_ep01(dataset: dict) -> dict:
    train, test, y, complete = _prepare(dataset)
    tau, theta = _fit(train, y, complete)
    log_tau = np.log(tau)
    times = np.asarray(dataset['eval_times'])
    weights = _rate_weights(y - train.matrix(log_tau) @ theta, complete, times)
    panel = dataset['test']
    initial = np.asarray(panel['strain_0'])
    test_complete = np.isfinite(panel['coupon_features_raw']).all(axis=1)
    pred = test.matrix(log_tau) @ theta
    if 'instant_rate_obs' in panel:
        pred = _update_drift(pred, panel['instant_rate_obs'], test_complete, times, weights)
    delta = pred[:, :-1]
    curves = initial[:, None] + delta
    return {
        'estimated_taus': tau,
        'predicted_strain': curves,
        'secant_compliance': (curves[:, -1] - initial) / np.maximum(np.abs(test.hold), 0.5),
        'hazard_crossing_times': _crossing_times(initial, curves, panel['critical_strain'], times),
    }
