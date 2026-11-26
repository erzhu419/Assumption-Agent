"""Calibrated two-branch hereditary creep for fresh, sub-yield coupons."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']])
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for lab in np.unique(labs):
        rows = labs == lab
        r = refs[rows]
        offset = np.nanmedian(np.nanmean(r, axis=0))
        gain = np.nanmedian(np.nanstd(r, axis=0, ddof=0))
        x[rows] = (x[rows] - offset) / gain
    cells = np.unique(np.column_stack([labs, strata]), axis=0)
    fill = np.nanmean([np.nanmean(x[(labs == l) & (strata == s)], axis=0)
                       for l, s in cells], axis=0)
    observed = np.isfinite(x)
    return np.where(observed, x, fill), observed


def _calibrate(cal):
    laws = np.empty((4, 2))
    for lab in range(4):
        rows = cal['lab_id'] == lab
        cmd = cal['cmd_stress'][rows]
        laws[lab] = np.linalg.lstsq(np.column_stack([cmd, np.ones(len(cmd))]),
                                   cal['true_stress'][rows], rcond=None)[0]
    return laws


class _Design:
    def __init__(self, panel, laws, dt, times, x):
        gain, bias = laws[panel['lab_id']].T
        self.past = panel['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.hold = panel['stress_hold_cmd'] * gain + bias
        self.dt = dt
        self.tail = np.cumsum(dt[::-1])[::-1] - dt
        self.times = times
        features = np.column_stack([np.ones(len(x)), x])
        self.linear = features[:, None, :] * np.r_[times, 1.0][None, :, None]

    def matrix(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u = self.dt[:, None] / tau
        v = self.tail[:, None] / tau
        h = self.times[:, None] / tau
        decay = np.exp(-v)
        kernel = decay * -np.expm1(-u)
        gap = self.hold[:, None] - self.past @ kernel
        response = np.vstack([-np.expm1(-h), 1.0 / tau])
        a = np.concatenate([gap[:, None, :] * response, self.linear], axis=2)
        if not derivative:
            return a
        dg = -self.past @ (v * kernel - u * decay * np.exp(-u))
        dr = np.vstack([-h * np.exp(-h), -1.0 / tau])
        return a, dg[:, None, :] * response + gap[:, None, :] * dr


def _scale(r):
    def mad(v):
        return max(1e-8, 1.4826 * np.median(np.abs(v - np.median(v))))
    return np.r_[np.full(r.shape[1] - 1, mad(r[:, :-1])), mad(r[:, -1])]


def _fit(design, y, rows):
    y = y[rows]
    low, high = np.log([0.051, 24.9])
    sep = 0.01

    def decode(p):
        return np.array([p[0], p[0] + sep + p[1] * (high - p[0] - sep)])

    grid = np.geomspace(0.085, 19.0, 11)
    starts = []
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            l1, l2 = np.log([t1, t2])
            a = design.matrix([l1, l2])[rows]
            w = np.linalg.lstsq(a.reshape(-1, 7), y.ravel(), rcond=None)[0]
            r = a @ w - y
            starts.append((np.sum(r*r), np.r_[l1, (l2-l1-sep)/(high-l1-sep), w], r))
    starts.sort(key=lambda item: item[0])
    bounds = (np.r_[low, 0., np.full(7, -np.inf)],
              np.r_[high-2*sep, 1., np.full(7, np.inf)])

    def optimize(p, scale, loss, cutoff):
        def fun(q):
            return ((design.matrix(decode(q))[rows] @ q[2:] - y) / scale).ravel()

        def jac(q):
            a, d = design.matrix(decode(q), derivative=True)
            d = d[rows] * q[2:4]
            d[:, :, 0] += (1 - q[1]) * d[:, :, 1]
            d[:, :, 1] *= high - q[0] - sep
            return (np.concatenate([d, a[rows]], axis=2) / scale[None, :, None]).reshape(-1, 9)

        return least_squares(fun, p, jac=jac, bounds=bounds, loss=loss,
                             f_scale=cutoff, max_nfev=150, x_scale='jac')

    scale = _scale(starts[0][2])
    pilots = [optimize(p, scale, 'soft_l1', 1.5) for _, p, _ in starts[:3]]
    p = min(pilots, key=lambda f: f.cost).x
    scale = _scale(design.matrix(decode(p))[rows] @ p[2:] - y)
    fit = min((optimize(f.x, scale, 'cauchy', 2.0) for f in pilots), key=lambda f: f.cost)
    return np.exp(decode(fit.x)), fit.x[2:]


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    x, seen = _features(train, test)
    laws = _calibrate(dataset['calibration_load_cells'])
    args = laws, dataset['past_dt'], dataset['eval_times']
    tr = _Design(train, *args, x[:n])
    te = _Design(test, *args, x[n:])
    y = np.column_stack([train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']])
    return tr, te, y, seen


def solve_alpha_ep04(dataset: dict) -> dict:
    tr, te, y, seen = _prepare(dataset)
    complete = seen.all(axis=1)
    tau, w = _fit(tr, y, complete[:len(y)])
    pred = te.matrix(np.log(tau)) @ w
    measured = dataset['test']['instant_rate_obs']
    innovation = np.where(np.isfinite(measured), measured - pred[:, -1], 0.0)
    # Use the measured rate to recover drift only when a feature was imputed.
    correction = np.where(complete[len(y):], 0.0, innovation)
    return {'estimated_taus': tau, 'hereditary_weights': w,
            'predicted_strain': dataset['test']['strain_0'][:, None] + pred[:, :-1]
                                + correction[:, None] * te.times,
            'initial_rate_pred': pred[:, -1] + correction}
