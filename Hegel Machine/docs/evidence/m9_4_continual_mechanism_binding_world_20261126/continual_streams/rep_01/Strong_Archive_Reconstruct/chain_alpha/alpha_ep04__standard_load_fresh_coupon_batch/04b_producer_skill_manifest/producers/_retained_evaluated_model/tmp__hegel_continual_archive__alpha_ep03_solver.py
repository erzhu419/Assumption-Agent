"""Calibrated two-branch creep with signed overstress and plastic-work memory."""
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
        self.dt = np.asarray(dt, dtype=float)
        self.tail = np.cumsum(self.dt[::-1])[::-1] - self.dt
        self.times = np.asarray(times)
        self.time_linear = np.r_[self.times, 1.0]
        self.ov = np.sign(hold) * np.maximum(np.abs(hold) - yield_stress, 0.0)**2
        self.work = (np.maximum(np.abs(past) - yield_stress, 0.0)**2 @ self.dt) * hold
        features = np.column_stack([np.ones(len(x)), x, self.ov])
        self.linear = features[:, None, :] * self.time_linear[None, :, None]

    def matrix(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u = self.dt[:, None] / tau
        v = self.tail[:, None] / tau
        h = self.times[:, None] / tau
        decay = np.exp(-v)
        w = decay * -np.expm1(-u)
        gap = self.hold[:, None] - self.past @ w
        basis = np.vstack([-np.expm1(-h), 1.0 / tau])
        memory = gap[:, None, :] * basis[None, :, :]
        plastic = self.work[:, None] * basis[None, :, 0]
        a = np.concatenate([memory, self.linear, plastic[..., None]], axis=2)
        if not derivative:
            return a
        dgap = -self.past @ (v * w - u * np.exp(-u) * decay)
        dbasis = np.vstack([-h * np.exp(-h), -1.0 / tau])
        dmemory = dgap[:, None, :] * basis[None, :, :] + gap[:, None, :] * dbasis[None, :, :]
        return a, dmemory, self.work[:, None] * dbasis[None, :, 0]


def _fit(design, y, rows):
    y = y[rows]
    pwidth = 9
    lmin, lmax = np.log([0.051, 24.9])
    min_gap = 0.005
    grid = np.linspace(lmin + 0.003, lmax - 0.003, 10)
    starts = []
    for i, l1 in enumerate(grid):
        for l2 in grid[i+1:]:
            a = design.matrix([l1, l2])[rows]
            coef = np.linalg.lstsq(a.reshape(-1, pwidth), y.ravel(), rcond=None)[0]
            r = a @ coef - y
            fraction = (l2 - l1 - min_gap) / (lmax - l1 - min_gap)
            starts.append((np.sum(r**2), np.r_[l1, fraction, coef], r))
    starts.sort(key=lambda z: z[0])

    def unpack(p):
        return np.array([p[0], p[0] + min_gap + p[1] * (lmax - p[0] - min_gap)])

    def noise(r):
        def mad(x):
            return max(1e-7, 1.4826 * np.median(np.abs(x - np.median(x))))
        return np.r_[np.full(r.shape[1] - 1, mad(r[:, :-1])), mad(r[:, -1])]

    def optimize(p, scale, loss, cutoff):
        def residual(p):
            return ((design.matrix(unpack(p))[rows] @ p[2:] - y) / scale).ravel()

        def jac(p):
            a, dm, dp = design.matrix(unpack(p), derivative=True)
            first = dm[..., 0] * p[2] + dp * p[-1]
            second = dm[..., 1] * p[3]
            d1 = first + (1.0 - p[1]) * second
            d2 = (lmax - p[0] - min_gap) * second
            j = np.concatenate([d1[..., None], d2[..., None], a], axis=2)[rows]
            return (j / scale[:, None]).reshape(-1, pwidth + 2)

        bounds = (np.r_[lmin, 0., np.full(pwidth, -np.inf)],
                  np.r_[lmax - min_gap - 1e-4, 1., np.full(pwidth, np.inf)])
        return least_squares(residual, p, jac=jac, bounds=bounds, loss=loss,
                             f_scale=cutoff, x_scale='jac', max_nfev=180)

    scale = noise(starts[0][2])
    pilots = [optimize(s[1], scale, 'soft_l1', 1.5) for s in starts[:3]]
    best = min(pilots, key=lambda result: result.cost)
    scale = noise(design.matrix(unpack(best.x))[rows] @ best.x[2:] - y)
    best = min((optimize(p.x, scale, 'cauchy', 2.0) for p in pilots), key=lambda result: result.cost)
    return unpack(best.x), best.x[2:]


def _prepare(dataset):
    x, complete = _aligned_panels(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times, threshold = dataset['past_dt'], dataset['eval_times'], dataset['yield_stress']
    train_design = _Design(*_physical_stresses(dataset, train), dt, times, x[:n], threshold)
    test_design = _Design(*_physical_stresses(dataset, test), dt, times, x[n:], threshold)
    y = np.column_stack([train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']])
    return train_design, test_design, y, complete


def solve_alpha_ep03(dataset: dict) -> dict:
    train_design, test_design, y, complete = _prepare(dataset)
    log_tau, theta = _fit(train_design, y, complete[:len(y)])
    predicted = test_design.matrix(log_tau) @ theta
    test = dataset['test']
    rate_error = test['instant_rate_obs'] - predicted[:, -1]
    # Use the rate innovation to recover drift when coupon features are missing.
    correction = np.where(complete[len(y):], 0.0, rate_error)
    strain = predicted[:, :-1] + correction[:, None] * test_design.times
    return {'estimated_taus': np.exp(log_tau), 'viscoplastic_weights': theta[7:9],
            'predicted_strain': strain + test['strain_0'][:, None]}
