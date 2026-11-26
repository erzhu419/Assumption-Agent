"""Load-cell calibration and preloaded two-branch coupon creep."""
import numpy as np
from scipy.optimize import least_squares


def _aligned(dataset):
    train, test = dataset['train'], dataset['test']
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    ref = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for lab in np.unique(labs):
        rows = labs == lab
        r = ref[rows]
        r = r[:, np.isfinite(r).any(axis=0)]
        x[rows] = (x[rows] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0, ddof=0))
    complete = np.isfinite(x).all(axis=1)
    _, cells = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = []
    for cell in np.unique(cells):
        block = x[cells == cell]
        count = np.isfinite(block).sum(axis=0)
        means.append(np.divide(np.nansum(block, axis=0), count, out=np.full(x.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0)), complete


def _physical(dataset, panel):
    cal = dataset['calibration_load_cells']
    past = np.empty_like(panel['past_stress_cmd'], dtype=float)
    hold = np.empty_like(panel['stress_hold_cmd'], dtype=float)
    labs = np.asarray(panel['lab_id'])
    for lab in np.unique(labs):
        c = np.asarray(cal['lab_id']) == lab
        cmd = np.asarray(cal['cmd_stress'])[c]
        gain, bias = np.linalg.lstsq(np.column_stack((cmd, np.ones_like(cmd))), np.asarray(cal['true_stress'])[c], rcond=None)[0]
        rows = labs == lab
        past[rows] = gain * np.asarray(panel['past_stress_cmd'])[rows] + bias
        hold[rows] = gain * np.asarray(panel['stress_hold_cmd'])[rows] + bias
    return past, hold


class _Design:
    def __init__(self, past, hold, dt, times, x, initial):
        self.past, self.hold, self.initial = past, hold, initial
        self.dt, self.times = np.asarray(dt), np.asarray(times)
        self.tail = np.cumsum(self.dt[::-1])[::-1] - self.dt
        self.drift = np.column_stack((np.ones(len(x)), x))[:, None, :] * np.r_[times, 1.0][None, :, None]

    def matrix(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u = self.dt[:, None] / tau
        v = self.tail[:, None] / tau
        h = self.times[:, None] / tau
        weights = np.exp(-v) * -np.expm1(-u)
        preload = self.initial * np.exp(-self.dt.sum() / tau)
        gap = self.hold[:, None] - self.past @ weights - preload
        basis = np.vstack((-np.expm1(-h), 1 / tau))
        a = np.concatenate((gap[:, None, :] * basis, self.drift), axis=2)
        if not derivative:
            return a
        dg = -(self.past @ (v * weights - u * np.exp(-u - v))) - preload * self.dt.sum() / tau
        db = np.vstack((-h * np.exp(-h), -1 / tau))
        return a, dg[:, None, :] * basis + gap[:, None, :] * db


def _mad(r):
    return max(1e-8, 1.4826 * np.median(np.abs(r - np.median(r))))


def _fit(design, y, rows):
    y = y[rows]
    ncoef = design.drift.shape[-1] + 2
    lmin, lmax = np.log([0.051, 24.9])
    grid = np.unique(np.r_[np.linspace(lmin, lmax, 10), np.log([0.25, 0.45, 0.7, 1.3, 3, 5, 9])])
    starts = []
    for i, l1 in enumerate(grid[:-1]):
        for l2 in grid[i+1:]:
            a = design.matrix([l1, l2])[rows]
            b = np.linalg.lstsq(a.reshape(-1, ncoef), y.ravel(), rcond=None)[0]
            starts.append((np.linalg.norm(a @ b - y), np.r_[l1, l2, b]))
    starts.sort(key=lambda z: z[0])
    bounds = (np.r_[[lmin]*2, [-np.inf]*ncoef], np.r_[[lmax]*2, [np.inf]*ncoef])

    def optimize(p, scale, loss):
        def fun(p):
            return ((design.matrix(p[:2])[rows] @ p[2:] - y) / scale).ravel()

        def jac(p):
            a, da = design.matrix(p[:2], derivative=True)
            return (np.concatenate((da[rows] * p[2:4], a[rows]), axis=2) / scale[None, :, None]).reshape(-1, ncoef + 2)

        return least_squares(fun, p, jac=jac, bounds=bounds, loss=loss, f_scale=2.5, x_scale='jac', max_nfev=120)

    fits = [optimize(p, np.ones(y.shape[1]), 'linear') for _, p in starts[:5]]
    ordered = [f for f in fits if f.x[0] < f.x[1]]
    p = min(ordered, key=lambda f: f.cost).x if ordered else starts[0][1]
    r = design.matrix(p[:2])[rows] @ p[2:] - y
    scale = np.r_[np.full(y.shape[1]-1, _mad(r[:, :-1].ravel())), _mad(r[:, -1])]
    fit = optimize(p, scale, 'soft_l1').x
    if fit[0] < fit[1]:
        p = fit
    return p[:2], p[2:]


def _lab_drift(train, times, residual, train_score, test_labs, test_score, complete):
    # Long-baseline differences cancel an additive strain measurement offset.
    j, k = np.triu_indices(len(times), 1)
    span = times[k] - times[j]
    across_hold = span > 1.5
    j, k, span = j[across_hold], k[across_hold], span[across_hold]
    correction = np.zeros(len(test_labs))
    for lab in np.unique(test_labs):
        rows = (train['lab_id'] == lab) & complete
        if not rows.any():
            continue
        r = residual[rows]
        slope = np.median((r[:, k] - r[:, j]) / span, axis=1)
        center = np.median(train_score[rows])
        median_slope = np.median(slope)
        design = np.column_stack((np.ones(rows.sum()), train_score[rows] - center))
        bias, scale = np.linalg.lstsq(design, np.clip(slope, median_slope - 0.04, median_slope + 0.04), rcond=None)[0]
        test_rows = test_labs == lab
        correction[test_rows] = bias + scale * (test_score[test_rows] - center)
    return correction


def _crossing(times, initial, strain, critical):
    t = np.r_[0.0, times]
    path = np.column_stack((initial, strain))
    hits = path >= critical[:, None]
    out = np.full(len(path), t[-1])
    out[hits[:, 0]] = 0.0
    rows = np.flatnonzero(~hits[:, 0] & hits.any(axis=1))
    j = hits[rows].argmax(axis=1)
    prev, end = path[rows, j-1], path[rows, j]
    out[rows] = t[j-1] + (t[j] - t[j-1]) * (critical[rows] - prev) / (end - prev)
    return out


def solve_alpha_ep01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    times = np.asarray(dataset['eval_times'])
    x, complete = _aligned(dataset)
    a, b = (_Design(*_physical(dataset, panel), dataset['past_dt'], times, features,
                    np.asarray(panel['initial_memory_states']))
            for panel, features in ((train, x[:n]), (test, x[n:])))
    y = np.column_stack((train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None], train['instant_rate_obs']))
    log_tau, weights = _fit(a, y, complete[:n])
    pred = b.matrix(log_tau) @ weights
    # The measured initial rate can refine the linear drift without altering
    # the fitted relaxation branches. Missing features weaken the drift prior.
    rate = np.asarray(test['instant_rate_obs'])
    innovation = np.where(np.isfinite(rate), rate - pred[:, -1], 0.0)
    rate_weight = np.where(complete[n:], 0.15, 1.0)
    residual = y[:, :-1] - (a.matrix(log_tau) @ weights)[:, :-1]
    feature_score = x @ weights[3:]
    lab_update = _lab_drift(train, times, residual, feature_score[:n],
                            np.asarray(test['lab_id']), feature_score[n:], complete[:n])
    drift_update = np.where(complete[n:], rate_weight * innovation + (1 - rate_weight) * lab_update, 0.95 * innovation)
    change = pred[:, :-1] + drift_update[:, None] * times
    initial = np.asarray(test['strain_0'])
    strain = initial[:, None] + change
    return {'estimated_taus': np.exp(log_tau), 'predicted_strain': strain,
            'secant_compliance': (strain[:, -1] - initial) / np.maximum(np.abs(b.hold), 0.5),
            'hazard_crossing_times': _crossing(times, initial, strain, np.asarray(test['critical_strain']))}
