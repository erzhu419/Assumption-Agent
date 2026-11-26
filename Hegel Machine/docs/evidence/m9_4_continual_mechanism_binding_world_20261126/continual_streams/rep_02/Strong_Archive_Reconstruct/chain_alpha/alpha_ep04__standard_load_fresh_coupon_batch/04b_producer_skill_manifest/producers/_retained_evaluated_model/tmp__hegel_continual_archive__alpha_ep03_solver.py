"""Calibrated two-branch hereditary creep with above-yield loading terms."""
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
    def __init__(self, past, hold, dt, times, x, yield_stress):
        self.past, self.hold, self.dt, self.times = past, hold, dt, times
        self.tail = np.cumsum(dt[::-1])[::-1] - dt
        self.linear_time = np.r_[times, 1.0]
        overstress = np.maximum(np.abs(hold) - yield_stress, 0.0)**2 * np.sign(hold)
        work = np.maximum(np.abs(past) - yield_stress, 0.0)**2 @ dt
        self.plastic_memory = work * hold
        linear = np.column_stack([np.ones(len(x)), x, overstress])
        self.linear = linear[:, None, :] * self.linear_time[None, :, None]

    def matrix(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u, v, h = self.dt[:, None] / tau, self.tail[:, None] / tau, self.times[:, None] / tau
        decay = np.exp(-v)
        weights = decay * -np.expm1(-u)
        gap = self.hold[:, None] - self.past @ weights
        basis = np.vstack([-np.expm1(-h), 1 / tau])
        memory = gap[:, None, :] * basis[None, :, :]
        plastic = self.plastic_memory[:, None, None] * basis[None, :, :1]
        a = np.concatenate([memory, self.linear, plastic], axis=2)
        if not derivative:
            return a
        dgap = -self.past @ (v * weights - u * decay * np.exp(-u))
        dbasis = np.vstack([-h * np.exp(-h), -1 / tau])
        dm = dgap[:, None, :] * basis + gap[:, None, :] * dbasis
        dp = self.plastic_memory[:, None] * dbasis[None, :, 0]
        return a, dm, dp


def _noise(r):
    def mad(v):
        return max(1e-8, 1.4826 * np.median(np.abs(v - np.median(v))))
    return np.r_[np.repeat(mad(r[:, :-1]), r.shape[1]-1), mad(r[:, -1])]


def _fit(design, y, rows, plastic=True):
    y = y[rows]
    cols = 9 if plastic else 7
    low, high = np.log([0.05001, 24.999])
    separation = 1e-5

    # Ordered log-times: the plastic history always uses the fast mode.
    def encode(l):
        return np.array([l[0], (l[1] - l[0] - separation)/(high - l[0] - separation)])

    def decode(p):
        return np.array([p[0], p[0] + separation + p[1]*(high - p[0] - separation)])

    grid = np.geomspace(0.085, 19.0, 11)
    starts = []
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            l = np.log([t1, t2])
            a = design.matrix(l)[rows, :, :cols]
            beta = np.linalg.lstsq(a.reshape(-1, cols), y.ravel(), rcond=None)[0]
            r = a @ beta - y
            starts.append((np.sum(r*r), np.r_[encode(l), beta], r))
    starts.sort(key=lambda s: s[0])
    bounds = (np.r_[low, 0.0, np.full(cols, -np.inf)],
              np.r_[high-2*separation, 1.0, np.full(cols, np.inf)])

    def optimize(p, scale, loss, cutoff):
        def fun(q):
            a = design.matrix(decode(q))[rows, :, :cols]
            return ((a @ q[2:] - y) / scale).ravel()

        def jac(q):
            a, dm, dp = design.matrix(decode(q), derivative=True)
            da = dm[rows] * q[2:4]
            if plastic:
                da[:, :, 0] += dp[rows] * q[-1]
            da[:, :, 0] += (1 - q[1]) * da[:, :, 1]
            da[:, :, 1] *= high - q[0] - separation
            return (np.concatenate([da, a[rows, :, :cols]], axis=2) / scale[None, :, None]).reshape(-1, cols+2)

        return least_squares(fun, p, jac=jac, bounds=bounds, loss=loss, f_scale=cutoff,
                             max_nfev=150, x_scale='jac')

    scale = _noise(starts[0][2])
    pilots = [optimize(s[1], scale, 'soft_l1', 1.5) for s in starts[:3]]
    p = min(pilots, key=lambda f: f.cost).x
    scale = _noise(design.matrix(decode(p))[rows, :, :cols] @ p[2:] - y)
    fit = min((optimize(f.x, scale, 'cauchy', 2.0) for f in pilots), key=lambda f: f.cost)
    return np.exp(decode(fit.x)), fit.x[2:]


def _prepare(dataset):
    x, complete = _aligned_panels(dataset)
    train, test = dataset['train'], dataset['test']
    args = dataset['past_dt'], dataset['eval_times']
    n = len(train['lab_id'])
    dtrain = _Design(*_physical_stresses(dataset, train), *args, x[:n], dataset['yield_stress'])
    dtest = _Design(*_physical_stresses(dataset, test), *args, x[n:], dataset['yield_stress'])
    y = np.column_stack([train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']])
    return dtrain, dtest, y, complete


def solve_alpha_ep03(dataset: dict) -> dict:
    train, test, y, complete = _prepare(dataset)
    tau, theta = _fit(train, y, complete[:len(y)])
    pred = test.matrix(np.log(tau)) @ theta
    rate = dataset['test']['instant_rate_obs']
    innovation = np.where(np.isfinite(rate), rate-pred[:, -1], 0.0)
    # With measured features, shrink the noisy rate innovation toward the model.
    # With imputed features, use the measured rate to recover the missing drift.
    drift = np.where(complete[len(y):], 0.25, 1.0) * innovation
    strain = pred[:, :-1] + drift[:, None]*test.times + dataset['test']['strain_0'][:, None]
    return {'estimated_taus': tau, 'viscoplastic_weights': theta[-2:], 'predicted_strain': strain}
