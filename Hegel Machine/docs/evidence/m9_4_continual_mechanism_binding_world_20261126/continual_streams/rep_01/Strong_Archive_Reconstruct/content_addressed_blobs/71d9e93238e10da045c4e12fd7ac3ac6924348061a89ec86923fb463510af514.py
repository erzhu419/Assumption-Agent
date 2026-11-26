"""Two-branch hereditary creep for the calibrated, sub-yield fresh batch."""
import numpy as np
from scipy.optimize import least_squares


def _prepare(dataset):
    train, test = (dataset[k] for k in ('train', 'test'))
    def joint(key):
        return np.concatenate((train[key], test[key]), axis=0)

    labs, strata = joint('lab_id'), joint('stratum')
    refs = joint('ref_coupons')
    x = joint('coupon_features_raw').astype(float)
    command = joint('past_stress_cmd')
    hold_cmd = joint('stress_hold_cmd')
    past, hold = np.empty_like(command, dtype=float), np.empty_like(hold_cmd, dtype=float)
    cal = dataset['calibration_load_cells']
    for lab in np.unique(labs):
        rows = labs == lab
        ref = refs[rows]
        offset = np.nanmedian(np.nanmean(ref, axis=0))
        gain = np.nanmedian(np.nanstd(ref, axis=0, ddof=0))
        x[rows] = (x[rows] - offset) / gain

        rows_cal = np.asarray(cal['lab_id']) == lab
        cmd = np.asarray(cal['cmd_stress'])[rows_cal]
        obs = np.asarray(cal['true_stress'])[rows_cal]
        gain, bias = np.linalg.lstsq(np.column_stack((cmd, np.ones_like(cmd))), obs, rcond=None)[0]
        past[rows], hold[rows] = gain * command[rows] + bias, gain * hold_cmd[rows] + bias

    _, cells = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    complete = np.isfinite(x).all(axis=1)
    cell_means = np.array([np.nanmean(x[cells == i], axis=0) for i in np.unique(cells)])
    x = np.where(np.isfinite(x), x, np.nanmean(cell_means, axis=0))
    return past, hold, x, complete


class _Design:
    def __init__(self, past, hold, dt, times, x):
        self.past = past
        self.hold = hold
        self.dt = np.asarray(dt)
        self.tail = np.cumsum(self.dt[::-1])[::-1] - self.dt
        self.times = np.asarray(times)
        features = np.column_stack((np.ones(len(x)), x))
        self.drift = features[:, None, :] * np.r_[self.times, 1.0][None, :, None]

    def __call__(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u, v, h = (vec[:, None] / tau for vec in (self.dt, self.tail, self.times))
        decay = np.exp(-v)
        history = decay * -np.expm1(-u)
        gap = self.hold[:, None] - self.past @ history
        response = np.vstack((-np.expm1(-h), 1.0 / tau))
        memory = gap[:, None, :] * response[None, :, :]
        a = np.concatenate((memory, self.drift), axis=2)
        if not derivative:
            return a
        dg = -self.past @ (v * history - u * np.exp(-u) * decay)
        dr = np.vstack((-h * np.exp(-h), -1.0 / tau))
        return a, dg[:, None, :] * response[None, :, :] + gap[:, None, :] * dr[None, :, :]


def _fit(design, y):
    lo, hi = np.log([0.0501, 24.99])
    gap = 0.004
    grid = np.linspace(lo + 0.01, hi - 0.01, 12)
    starts = []
    for i, u in enumerate(grid):
        for v in grid[i + 1:]:
            a = design([u, v])
            w = np.linalg.lstsq(a.reshape(-1, 7), y.ravel(), rcond=None)[0]
            error = a @ w - y
            starts.append((np.sum(error**2), np.r_[u, (v-u-gap)/(hi-u-gap), w]))

    def logs(p):
        return np.array([p[0], p[0] + gap + p[1] * (hi - p[0] - gap)])

    def optimize(p, scale, loss):
        def fun(p):
            return ((design(logs(p)) @ p[2:] - y) / scale).ravel()

        def jac(p):
            a, d = design(logs(p), derivative=True)
            first, second = np.moveaxis(d * p[2:4], -1, 0)
            j = np.concatenate(((first + (1-p[1])*second)[..., None],
                                ((hi-p[0]-gap)*second)[..., None], a), axis=2)
            return (j / scale[..., None]).reshape(-1, 9)

        return least_squares(fun, p, jac=jac, loss=loss, f_scale=2.0,
                             bounds=(np.r_[lo, 0., np.full(7, -np.inf)],
                                     np.r_[hi-gap, 1., np.full(7, np.inf)]), max_nfev=170)

    ranked = sorted(starts, key=lambda pair: pair[0])[:4]
    baseline = min((optimize(p, np.ones((1,)), 'linear') for _, p in ranked), key=lambda r: r.cost)
    res = design(logs(baseline.x)) @ baseline.x[2:] - y
    def mad(x):
        return max(1e-7, 1.4826 * np.median(np.abs(x - np.median(x))))
    scale = np.r_[np.full(y.shape[1]-1, mad(res[:, :-1])), mad(res[:, -1])]
    robust = optimize(baseline.x, scale, 'soft_l1')
    return logs(robust.x), robust.x[2:]


def solve_alpha_ep04(dataset: dict) -> dict:
    train, test = (dataset[k] for k in ('train', 'test'))
    n = len(train['lab_id'])
    past, hold, features, complete = _prepare(dataset)
    dt, times = (dataset[k] for k in ('past_dt', 'eval_times'))
    rows = np.flatnonzero(complete[:n])
    y = np.column_stack((train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None],
                         train['instant_rate_obs']))
    log_tau, w = _fit(_Design(past[rows], hold[rows], dt, times, features[rows]), y[rows])
    prediction = _Design(past[n:], hold[n:], dt, times, features[n:])(log_tau) @ w
    # When features are missing, recover the coupon's drift from its measured
    # onset rate minus the fitted hereditary contribution and prior drift.
    imputed = ~complete[n:]
    innovation = np.asarray(test['instant_rate_obs'])[imputed] - prediction[imputed, -1]
    prediction[imputed] += innovation[:, None] * np.r_[times, 1.0]
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': w,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + prediction[:, :-1],
            'initial_rate_pred': prediction[:, -1]}
