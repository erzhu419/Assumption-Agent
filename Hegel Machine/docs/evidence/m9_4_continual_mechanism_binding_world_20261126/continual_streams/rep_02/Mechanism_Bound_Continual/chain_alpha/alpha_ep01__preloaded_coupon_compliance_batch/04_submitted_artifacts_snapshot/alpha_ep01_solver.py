"""Calibrated two-branch hereditary creep with measured initial memory."""

import numpy as np
from scipy.optimize import least_squares


def _align(dataset):
    train, test = dataset['train'], dataset['test']
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    complete = np.isfinite(x).all(axis=1)
    for lab in np.unique(labs):
        rows = labs == lab
        r = ref[rows]
        offset = np.nanmedian(np.nanmean(r, axis=0))
        gain = np.nanmedian(np.nanstd(r, axis=0, ddof=0))
        x[rows] = (x[rows] - offset) / gain
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = [np.nanmean(x[cell == c], axis=0) for c in np.unique(cell)]
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0)), complete


def _calibrate(cal):
    gain, bias = np.empty(4), np.empty(4)
    for lab in range(4):
        rows = cal['lab_id'] == lab
        cmd = cal['cmd_stress'][rows]
        gain[lab], bias[lab] = np.linalg.lstsq(
            np.column_stack((cmd, np.ones_like(cmd))), cal['true_stress'][rows], rcond=None
        )[0]
    return gain, bias


class _Design:
    def __init__(self, panel, x, gain, bias, dt, times, rows=slice(None)):
        lab = panel['lab_id'][rows]
        self.past = gain[lab, None] * panel['past_stress_cmd'][rows] + bias[lab, None]
        self.hold = gain[lab] * panel['stress_hold_cmd'][rows] + bias[lab]
        self.initial = panel['initial_memory_states'][rows]
        self.dt = dt[:, None]
        self.tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0][:, None]
        self.total = dt.sum()
        self.times = times[:, None]
        f = np.column_stack((np.ones(len(x[rows])), x[rows]))
        self.drift = f[:, None, :] * np.r_[times, 1.0][None, :, None]

    def matrix(self, log_tau, jac=False):
        tau = np.exp(log_tau)
        u, v, h = self.dt / tau, self.tail / tau, self.times / tau
        decay = np.exp(-v)
        weights = decay * -np.expm1(-u)
        preload = self.initial * np.exp(-self.total / tau)
        gap = self.hold[:, None] - self.past @ weights - preload
        time_basis = np.vstack((-np.expm1(-h), 1.0 / tau))
        a = np.concatenate((gap[:, None, :] * time_basis, self.drift), axis=2)
        if not jac:
            return a
        dgap = -(self.past @ (v * weights - u * np.exp(-u) * decay)) - preload * (self.total / tau)
        dtime = np.vstack((-h * np.exp(-h), -1.0 / tau))
        return a, dgap[:, None, :] * time_basis + gap[:, None, :] * dtime


def _noise(r):
    def mad(a):
        return max(1e-8, 1.4826 * np.median(np.abs(a - np.median(a))))
    return np.r_[np.full(r.shape[1] - 1, mad(r[:, :-1])), mad(r[:, -1])]


def _fit(design, y):
    grid = np.geomspace(0.15, 16.0, 9)
    starts = []
    for i, first in enumerate(grid):
        for second in grid[i + 1:]:
            l = np.log([first, second])
            a = design.matrix(l)
            c = np.linalg.lstsq(a.reshape(-1, 7), y.ravel(), rcond=None)[0]
            r = a @ c - y
            starts.append((np.sum(r * r), np.r_[l, c], r))
    starts.sort(key=lambda z: z[0])
    bounds = (np.r_[[np.log(0.05)] * 2, [-np.inf] * 7],
              np.r_[[np.log(25.0)] * 2, [np.inf] * 7])

    def optimize(p, scale, loss):
        def residual(q):
            return ((design.matrix(q[:2]) @ q[2:] - y) / scale).ravel()

        def jacobian(q):
            a, da = design.matrix(q[:2], jac=True)
            return (np.concatenate((da * q[2:4], a), axis=2) / scale[:, None]).reshape(-1, 9)

        return least_squares(residual, p, jac=jacobian, bounds=bounds, loss=loss,
                             f_scale=2.5, x_scale='jac', max_nfev=140)

    scale = _noise(starts[0][2])
    pilot = min((optimize(s[1], scale, 'soft_l1') for s in starts[:3]), key=lambda o: o.cost).x
    scale = _noise(design.matrix(pilot[:2]) @ pilot[2:] - y)
    result = optimize(pilot, scale, 'cauchy').x
    return result[:2], result[2:]


def _predict(design, log_tau, coef, panel, complete, times):
    pred = design.matrix(log_tau) @ coef
    # Initial rate updates the drift only; the imputed-feature prior is weaker.
    innovation = panel['instant_rate_obs'] - pred[:, -1]
    drift = np.where(complete, 0.20, 0.98) * innovation
    return panel['strain_0'][:, None] + pred[:, :-1] + drift[:, None] * times


def _crossings(strain_0, strain, critical, times):
    values = np.column_stack((strain_0, strain))
    reached = values >= critical[:, None]
    first = np.argmax(reached, axis=1)
    crossing = np.full(len(strain_0), times[-1])
    crossing[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    t = np.r_[0.0, times]
    v0, v1 = values[rows, j - 1], values[rows, j]
    crossing[rows] = t[j - 1] + (t[j] - t[j - 1]) * (critical[rows] - v0) / (v1 - v0)
    return crossing


def solve_alpha_ep01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    x, complete = _align(dataset)
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    fit_rows = complete[:n]
    dtrain = _Design(train, x[:n], gain, bias, dt, times, fit_rows)
    increments = train['strain_curves_obs'] - train['strain_0'][:, None]
    y = np.column_stack((increments, train['instant_rate_obs']))[fit_rows]
    log_tau, coef = _fit(dtrain, y)
    dtest = _Design(test, x[n:], gain, bias, dt, times)
    strain = _predict(dtest, log_tau, coef, test, complete[n:], times)
    return {
        'estimated_taus': np.sort(np.exp(log_tau)),
        'predicted_strain': strain,
        'secant_compliance': (strain[:, -1] - test['strain_0']) / np.maximum(np.abs(dtest.hold), 0.5),
        'hazard_crossing_times': _crossings(test['strain_0'], strain, test['critical_strain'], times),
    }
