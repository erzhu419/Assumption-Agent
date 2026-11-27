"""Calibrated sub-yield creep fit for the multi-wing coupon stream."""
import numpy as np
from scipy.optimize import least_squares


def _stress_calibration(checks):
    lab = np.asarray(checks['lab_id'])
    cmd = np.asarray(checks['cmd_stress'], float)
    true = np.asarray(checks['true_stress'], float)
    params = np.tile([1., 0.], (4, 1))
    for c in range(4):
        use = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[use], true[use]
        if not len(x):
            continue
        i, j = np.triu_indices(len(x), 1)
        distinct = abs(x[i] - x[j]) > 1e-8
        i, j = i[distinct], j[distinct]
        if not len(i):
            params[c, 1] = np.median(y - x)
            continue
        slopes = (y[i] - y[j]) / (x[i] - x[j])
        errors = y[None, :] - slopes[:, None] * x[None, :]
        intercepts = np.median(errors, axis=1)
        errors -= intercepts[:, None]
        best = np.argmin(np.median(abs(errors), axis=1))
        cutoff = max(7e-4, 4.5 * 1.4826 * np.median(abs(errors[best])))
        inliers = abs(errors[best]) <= cutoff
        params[c] = np.linalg.lstsq(np.column_stack((x[inliers], np.ones(sum(inliers)))), y[inliers], rcond=None)[0]
    return params


def _align(train, test):
    lab = np.concatenate((train['lab_id'], test['lab_id']))
    stratum = np.concatenate((train['stratum'], test['stratum']))
    x = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        offset = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0))
        x[use] = (x[use] - offset) / gain
    # Give each observed lab/stratum cell equal row mass before excluding missing values.
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[cell, None]
    valid = np.isfinite(x)
    mass = np.sum(w * valid, axis=0)
    fill = np.divide(np.sum(w * np.where(valid, x, 0.), axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0.)
    return np.where(valid, x, fill)


def _inputs(data, x, calibration):
    gain, bias = calibration[np.asarray(data['lab_id'])].T
    past = gain[:, None] * data['past_stress_cmd'] + bias[:, None]
    hold = gain * data['stress_hold_cmd'] + bias
    return past, hold, np.column_stack((np.ones(len(x)), x))


class _Design:
    def __init__(self, dt, t, inputs):
        self.dt, self.t = np.asarray(dt), np.asarray(t)
        self.past, self.hold, self.x = inputs
        self.base = np.empty((len(self.x), len(t) + 1, 7))
        self.base[:, :-1, 2:] = self.x[:, None, :] * self.t[None, :, None]
        self.base[:, -1, 2:] = self.x

    def make(self, taus, derivative=False):
        a = self.base.copy()
        if derivative:
            da = np.empty(a.shape[:2] + (2,))
        for m, tau in enumerate(taus):
            z = np.zeros(len(self.x))
            dz = np.zeros_like(z)
            e = np.exp(-self.dt / tau)
            f = -np.expm1(-self.dt / tau)
            de = e * self.dt / tau**2
            for j in range(len(e)):
                if derivative:
                    dz = e[j] * dz + de[j] * (z - self.past[:, j])
                z = e[j] * z + f[j] * self.past[:, j]
            gap = self.hold - z
            h = -np.expm1(-self.t / tau)
            a[:, :-1, m] = gap[:, None] * h
            a[:, -1, m] = gap / tau
            if derivative:
                dh = -np.exp(-self.t / tau) * self.t / tau**2
                da[:, :-1, m] = gap[:, None] * dh - dz[:, None] * h
                da[:, -1, m] = -dz / tau - gap / tau**2
        return (a, da) if derivative else a


def _fit(design, curves, initial_rate):
    y = np.column_stack((curves, initial_rate))
    def scales(r):
        def mad(a):
            return 1.4826 * np.median(abs(a - np.median(a)))
        s = max(1e-7, mad(r[:, :-1]))
        return np.r_[np.full(y.shape[1] - 1, s), max(1e-7, mad(r[:, -1]))]

    def linear_start(a):
        a = a.reshape(-1, 7)
        yflat = y.ravel()
        b = np.linalg.lstsq(a, yflat, rcond=None)[0]
        for _ in range(3):
            r = (np.einsum('ij,j->i', a, b) - yflat).reshape(y.shape)
            s = 2.5 * scales(r)
            w = (1. / np.hypot(s, r)).ravel()
            b = np.linalg.lstsq(a * w[:, None], yflat * w, rcond=None)[0]
        return b, (np.einsum('ij,j->i', a, b) - yflat).reshape(y.shape)

    seeds = []
    grid = (.22, .38, .68, 1.20, 2.15, 3.8, 6.5, 7.9)
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            b, r = linear_start(design.make((t1, t2)))
            seeds.append((np.r_[t1, t2, b], r))
    pilot = min(seeds, key=lambda sr: np.median(abs(sr[1][:, :-1])))
    scale = 2.5 * scales(pilot[1])
    bounds = (np.r_[.20, .20, np.full(7, -np.inf)], np.r_[8., 8., np.full(7, np.inf)])

    def score(r):
        return np.sum(np.log1p((r / scale)**2))

    def optimize(p):
        def residual(q):
            a = design.make(q[:2])
            return ((np.einsum('ntj,j->nt', a, q[2:]) - y) / scale).ravel()

        def jac(q):
            a, da = design.make(q[:2], derivative=True)
            j = np.concatenate((da * q[2:4], a), axis=2)
            return (j / scale[None, :, None]).reshape(-1, 9)

        return least_squares(residual, p, jac=jac, bounds=bounds, loss='cauchy',
                             tr_solver='exact', x_scale='jac', max_nfev=120,
                             ftol=2e-9, xtol=2e-9, gtol=2e-7)

    seeds.sort(key=lambda sr: score(sr[1]))
    fitted = min((optimize(p) for p, _ in seeds[:4]), key=lambda result: result.cost)
    scale = 2.5 * scales(fitted.fun.reshape(y.shape) * scale)
    p = optimize(fitted.x).x
    if p[0] > p[1]:
        p[[0, 1, 2, 3]] = p[[1, 0, 3, 2]]
    if p[0] == p[1]:
        if p[1] < 8.:
            p[1] = np.nextafter(p[1], np.inf)
        else:
            p[0] = np.nextafter(p[0], -np.inf)
    return p[:2], p[2:]


def solve_alpha_m96_w02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x = _align(train, test)
    n = len(train['lab_id'])
    cal = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    dtr = _Design(dt, t, _inputs(train, x[:n], cal))
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, theta = _fit(dtr, curves, train['instant_rate_obs'])
    dte = _Design(dt, t, _inputs(test, x[n:], cal))
    pred = np.einsum('ntj,j->nt', dte.make(tau), theta)
    # Missing features affect only the linear drift. The measured onset
    # identifies that drift after the hereditary onset has been accounted for.
    if 'instant_rate_obs' in test:
        onset = np.asarray(test['instant_rate_obs'])
        use = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(onset)
        drift_correction = onset[use] - pred[use, -1]
        pred[use, :-1] += drift_correction[:, None] * t
        pred[use, -1] += drift_correction
    return dict(estimated_taus=tau, hereditary_weights=theta,
                viscoplastic_weights=np.zeros(2),
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=pred[:, -1])
