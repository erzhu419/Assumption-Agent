"""Calibrated two-branch hereditary creep with high-load plasticity."""
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
    def __init__(self, past, hold, dt, times, features, yield_stress):
        self.past = past
        self.hold = hold
        self.dt = np.asarray(dt)
        self.tail = np.cumsum(self.dt[::-1])[::-1] - self.dt
        self.times = np.asarray(times)
        self.ramp = np.r_[self.times, 1.0]
        drift = np.column_stack((np.ones(len(hold)), features))
        self.linear = drift[:, None, :] * self.ramp[None, :, None]
        self.over = np.maximum(np.abs(hold) - yield_stress, 0)**2 * np.sign(hold)
        self.work = (np.maximum(np.abs(past) - yield_stress, 0)**2 @ self.dt) * hold

    def matrix(self, logs, derivative=False):
        tau = np.exp(logs)
        u = self.dt[:, None] / tau
        v = self.tail[:, None] / tau
        h = self.times[:, None] / tau
        decay = np.exp(-v)
        weights = decay * -np.expm1(-u)
        gap = self.hold[:, None] - self.past @ weights
        response = np.vstack((-np.expm1(-h), 1 / tau))
        memory = gap[:, None, :] * response
        flow = self.over[:, None] * self.ramp
        hardening = self.work[:, None] * response[:, 0]
        a = np.concatenate((memory, self.linear, flow[..., None], hardening[..., None]), axis=2)
        if not derivative:
            return a
        dg = -self.past @ (v * weights - u * np.exp(-u) * decay)
        dr = np.vstack((-h * np.exp(-h), -1 / tau))
        dm = dg[:, None, :] * response + gap[:, None, :] * dr
        dh = self.work[:, None] * dr[:, 0]
        return a, dm, dh


def _prepare(dataset):
    x, complete = _aligned_panels(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, t, sy = (dataset[k] for k in ('past_dt', 'eval_times', 'yield_stress'))
    dtrain = _Design(*_physical_stresses(dataset, train), dt, t, x[:n], sy)
    dtest = _Design(*_physical_stresses(dataset, test), dt, t, x[n:], sy)
    y = np.column_stack((train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None],
                         train['instant_rate_obs']))
    return dtrain, dtest, y, complete


_LOW, _HIGH = np.log([0.05001, 24.999])


def _tau_logs(q):
    # Keep the plastic-work response attached to the faster branch throughout fitting.
    return np.array([q[0], q[0] + (_HIGH - q[0]) * q[1]])


def _channel_scale(r):
    curve, rate = r[:, :-1].ravel(), r[:, -1]
    def mad(x):
        return max(1e-8, 1.4826 * np.median(np.abs(x - np.median(x))))
    return np.r_[np.full(r.shape[1] - 1, mad(curve)), mad(rate)]


def _fit(design, y, use):
    y = y[use]
    starts = []
    grid = np.geomspace(0.14, 13.0, 10)
    for i, t1 in enumerate(grid):
        for t2 in grid[i+1:]:
            l1, l2 = np.log([t1, t2])
            a = design.matrix([l1, l2])[use]
            coef = np.linalg.lstsq(a.reshape(-1, 9), y.ravel(), rcond=None)[0]
            residual = a @ coef - y
            starts.append((np.sum(residual**2), np.r_[l1, (l2-l1)/(_HIGH-l1), coef], residual))
    starts.sort(key=lambda s: s[0])

    def optimize(p, scale, loss, f_scale):
        def residual(q):
            return ((design.matrix(_tau_logs(q))[use] @ q[2:] - y) / scale).ravel()

        def jac(q):
            a, dm, dh = design.matrix(_tau_logs(q), derivative=True)
            d1 = dm[..., 0] * q[2] + dh * q[-1]
            d2 = dm[..., 1] * q[3]
            full = np.concatenate(((d1 + (1-q[1])*d2)[..., None],
                                   ((_HIGH-q[0])*d2)[..., None], a), axis=2)
            return (full[use] / scale[:, None]).reshape(-1, 11)

        return least_squares(residual, p, jac=jac, loss=loss, f_scale=f_scale,
                             bounds=(np.r_[_LOW, 1e-5, [-np.inf]*9],
                                     np.r_[_HIGH-1e-5, 1., [np.inf]*9]),
                             max_nfev=150, x_scale='jac')

    warm = _channel_scale(starts[0][2])
    pilots = [optimize(s[1], warm, 'soft_l1', 1.5) for s in starts[:3]]
    pilot = min(pilots, key=lambda fit: fit.cost).x
    scale = _channel_scale(design.matrix(_tau_logs(pilot))[use] @ pilot[2:] - y)
    result = min((optimize(f.x, scale, 'cauchy', 2.0) for f in pilots), key=lambda fit: fit.cost).x
    return _tau_logs(result), result[2:]


def solve_alpha_ep03(dataset: dict) -> dict:
    train, test, y, complete = _prepare(dataset)
    n = len(y)
    logs, theta = _fit(train, y, complete[:n])
    pred = test.matrix(logs) @ theta
    # The observed onset rate provides a coupon-specific update to linear drift.
    innovation = np.asarray(dataset['test']['instant_rate_obs']) - pred[:, -1]
    correction = np.where(complete[n:], 0.25, 1.0) * innovation
    strain = pred[:, :-1] + correction[:, None]*test.times + np.asarray(dataset['test']['strain_0'])[:, None]
    return {'estimated_taus': np.exp(logs), 'viscoplastic_weights': theta[-2:], 'predicted_strain': strain}
