"""Calibrated hereditary creep with signed hold overstress and plastic-work memory."""
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
        self.past, self.hold = past, hold
        self.dt, self.t = np.asarray(dt), np.asarray(times)
        self.tail = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.ov = np.sign(hold) * np.maximum(np.abs(hold) - yield_stress, 0.0)**2
        self.work = np.maximum(np.abs(past) - yield_stress, 0.0)**2 @ self.dt
        self.work_hold = self.work * hold
        self.linear_time = np.r_[self.t, 1.0]
        self.drift = np.c_[np.ones(len(x)), x][:, None, :] * self.linear_time[None, :, None]

    def matrix(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u, v, h = self.dt[:, None] / tau, self.tail[:, None] / tau, self.t[:, None] / tau
        decay = np.exp(-v)
        weights = decay * -np.expm1(-u)
        gap = self.hold[:, None] - self.past @ weights
        time_basis = np.vstack([-np.expm1(-h), 1 / tau])
        branches = gap[:, None, :] * time_basis[None, :, :]
        overload = self.ov[:, None] * self.linear_time
        plastic = self.work_hold[:, None] * time_basis[None, :, 0]
        a = np.concatenate([branches, self.drift, overload[..., None], plastic[..., None]], axis=2)
        if not derivative:
            return a
        dgap = -self.past @ (v * weights - u * np.exp(-u) * decay)
        dtime = np.vstack([-h * np.exp(-h), -1 / tau])
        db = dgap[:, None, :] * time_basis[None, :, :] + gap[:, None, :] * dtime[None, :, :]
        return a, db, self.work_hold[:, None] * dtime[None, :, 0]


def _scale(r):
    def mad(x):
        return max(1e-8, 1.4826 * np.median(np.abs(x - np.median(x))))
    return np.r_[np.full(r.shape[1] - 1, mad(r[:, :-1])), mad(r[:, -1])]


def _fit(design, y, rows, high_load=True):
    y = y[rows]
    ncoef = 9 if high_load else 7
    low, high = np.log([0.051, 24.9])
    sep = 1e-4

    def unpack(q):
        return np.array([q[0], q[0] + sep + q[1] * (high - q[0] - sep)])

    starts = []
    grid = np.log(np.geomspace(0.12, 16., 10))
    for i, a in enumerate(grid):
        for b in grid[i + 1:]:
            m = design.matrix([a, b])[rows, :, :ncoef]
            c = np.linalg.lstsq(m.reshape(-1, ncoef), y.ravel(), rcond=None)[0]
            r = m @ c - y
            starts.append((np.sum(r*r), np.r_[a, (b-a-sep)/(high-a-sep), c], r))
    starts.sort(key=lambda s: s[0])

    def optimize(p, scale, loss, f_scale):
        def residual(q):
            a = design.matrix(unpack(q))[rows, :, :ncoef]
            return ((a @ q[2:] - y) / scale).ravel()

        def jac(q):
            a, db, dp = design.matrix(unpack(q), derivative=True)
            d = db[rows] * q[2:4]
            if high_load:
                d[..., 0] += dp[rows] * q[-1]
            d[..., 0] += (1-q[1]) * d[..., 1]
            d[..., 1] *= high-q[0]-sep
            full = np.concatenate([d, a[rows, :, :ncoef]], axis=2)
            return (full / scale[None, :, None]).reshape(-1, ncoef+2)

        return least_squares(residual, p, jac=jac, loss=loss, f_scale=f_scale,
                             bounds=(np.r_[low, 0., [-np.inf]*ncoef],
                                     np.r_[high-sep, 1., [np.inf]*ncoef]),
                             x_scale='jac', max_nfev=180)

    warm_scale = _scale(starts[0][2])
    pilots = [optimize(s[1], warm_scale, 'soft_l1', 1.5) for s in starts[:3]]
    best = min(pilots, key=lambda f: f.cost).x
    scale = _scale(design.matrix(unpack(best))[rows, :, :ncoef] @ best[2:] - y)
    result = min((optimize(p.x, scale, 'cauchy', 2.) for p in pilots), key=lambda f: f.cost).x
    return unpack(result), result[2:]


def _prepare(dataset):
    x, complete = _aligned_panels(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times, threshold = (dataset[k] for k in ('past_dt', 'eval_times', 'yield_stress'))
    dtrain = _Design(*_physical_stresses(dataset, train), dt, times, x[:n], threshold)
    dtest = _Design(*_physical_stresses(dataset, test), dt, times, x[n:], threshold)
    y = np.column_stack([train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None],
                         train['instant_rate_obs']])
    return dtrain, dtest, y, complete


def _predict(design, log_tau, theta, rate, complete):
    p = design.matrix(log_tau)[..., :len(theta)] @ theta
    delta = np.where(np.isfinite(rate), rate - p[:, -1], 0.0)
    correction = np.where(complete, 0.25, 1.0) * delta
    return p[:, :-1] + correction[:, None] * design.t


def solve_alpha_ep03(dataset: dict) -> dict:
    dtrain, dtest, y, complete = _prepare(dataset)
    n = len(y)
    log_tau, theta = _fit(dtrain, y, complete[:n])
    test = dataset['test']
    pred = _predict(dtest, log_tau, theta, test['instant_rate_obs'], complete[n:])
    return {'estimated_taus': np.exp(log_tau), 'viscoplastic_weights': theta[-2:],
            'predicted_strain': pred + np.asarray(test['strain_0'])[:, None]}
