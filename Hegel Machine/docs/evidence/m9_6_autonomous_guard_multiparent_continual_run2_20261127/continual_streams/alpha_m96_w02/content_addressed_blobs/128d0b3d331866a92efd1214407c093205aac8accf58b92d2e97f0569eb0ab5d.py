"""Calibrate load cells and fit a two-branch hereditary creep law."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _prior():
    try:
        with open(_ARCHIVE) as f:
            records = json.load(f)
    except FileNotFoundError:
        return None
    recent = records[-1]
    lab, x, y = (np.asarray(recent[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    return np.array([_line(x[lab == c], y[lab == c]) for c in range(4)])


def _calibrate(checks):
    lab, cmd, true = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    params = np.empty((4, 2))
    prior = _prior()
    for c in range(4):
        keep = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[keep], true[keep]
        if len(x) >= 3 and np.ptp(x) >= .1:
            p = _line(x, y)
            if np.max(abs(y - (p[0] * x + p[1]))) <= 7e-4:
                params[c] = p
                continue
        p0 = prior[c] if prior is not None else np.array([1., 0.])
        params[c] = p0
        if len(x) <= 2 and prior is not None:
            # A singleton or pair cannot establish both a new affine map and
            # the absence of EMI. Only checks consistent with the latest sweep
            # can update that map without independent in-batch corroboration.
            consistent = abs(y - (p0[0] * x + p0[1])) <= .007
            x, y = x[consistent], y[consistent]
        if not len(x):
            continue
        i, j = np.triu_indices(len(x), 1)
        keep = abs(x[i] - x[j]) >= .1
        i, j = i[keep], j[keep]
        slopes = (y[i] - y[j]) / (x[i] - x[j])
        gains = np.r_[p0[0], slopes[slopes > 0.]]
        intercepts = np.median(y - gains[:, None] * x, axis=1)
        err = abs(y - gains[:, None] * x - intercepts[:, None])
        consensus = err <= 7e-4
        counts = consensus.sum(axis=1)
        # Two points alone cannot distinguish a clean pair from an EMI hit.
        # In tied sparse fits, prefer continuation of the archived affine map.
        deviations = (gains - p0[0])**2 + (intercepts - p0[1])**2
        order = np.lexsort((np.minimum(err, 7e-4).sum(axis=1), deviations, -counts))
        k = order[0]
        ok = consensus[k]
        if counts[k] >= 2 and np.ptp(x[ok]) >= .1:
            params[c] = _line(x[ok], y[ok])
        else:
            residual = y - p0[0] * x - p0[1]
            if len(x) == 1 or np.ptp(residual) <= .03:
                params[c, 1] += np.median(residual)
            else:
                params[c, 1] += residual[np.argmin(abs(residual))]
    return params


def _align(train, test):
    lab, stratum, raw, refs = (np.concatenate((train[k], test[k])) for k in
                               ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    x = raw.astype(float)
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (x[use] - center) / scale
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    mass = 1. / counts[cell, None]
    finite = np.isfinite(x)
    fill = np.sum(mass * np.where(finite, x, 0.), axis=0) / np.sum(mass * finite, axis=0)
    return np.where(finite, x, fill)


class _Design:
    def __init__(self, data, x, calibration, dt, time, yield_stress):
        gain, bias = calibration[data['lab_id']].T
        past = gain[:, None] * data['past_stress_cmd'] + bias[:, None]
        self.hold = gain * data['stress_hold_cmd'] + bias
        self.past = past
        self.dt = np.asarray(dt)
        self.lag = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.]
        self.time = np.asarray(time)
        self.over = np.maximum(abs(self.hold) - yield_stress, 0.)**2 * np.sign(self.hold)
        self.work = (np.maximum(abs(past) - yield_stress, 0.)**2 @ self.dt) * self.hold
        self.active = bool(np.any(self.over) or np.any(self.work))
        self.base = np.zeros((len(x), len(time) + 1, 9 if self.active else 7))
        drift = np.column_stack((np.ones(len(x)), x))
        self.base[:, :-1, 2:7] = drift[:, None, :] * self.time[None, :, None]
        self.base[:, -1, 2:7] = drift
        if self.active:
            self.base[:, :-1, 7] = self.over[:, None] * self.time
            self.base[:, -1, 7] = self.over

    def matrix(self, tau):
        a = self.base.copy()
        for j, q in enumerate(tau):
            kernel = np.exp(-self.lag / q) * -np.expm1(-self.dt / q)
            gap = self.hold - np.einsum('ij,j->i', self.past, kernel)
            relaxation = -np.expm1(-self.time / q)
            a[:, :-1, j] = gap[:, None] * relaxation
            a[:, -1, j] = gap / q
            if j == 0 and self.active:
                a[:, :-1, 8] = self.work[:, None] * relaxation
                a[:, -1, 8] = self.work / q
        return a


def _scale(r):
    def mad(x):
        return max(1e-7, 1.4826 * np.median(abs(x - np.median(x))))
    return np.r_[np.full(r.shape[1] - 1, mad(r[:, :-1])), mad(r[:, -1])]


def _fit(design, y):
    defaults = (.16, .32, .55, .9, 1.5, 2.6, 4.3, 7.5, 14.)
    seeds = []
    for i, fast in enumerate(defaults[:-1]):
        for slow in defaults[i+1:]:
            a = design.matrix((fast, slow)).reshape(y.size, -1)
            b = np.linalg.lstsq(a, y.ravel(), rcond=None)[0]
            for _ in range(3):
                r = (np.einsum('ij,j->i', a, b) - y.ravel()).reshape(y.shape)
                w = (1. / np.hypot(r, 2.5 * _scale(r))).ravel()
                w /= w.max()
                b = np.linalg.lstsq(a * w[:, None], y.ravel() * w, rcond=None)[0]
            r = (np.einsum('ij,j->i', a, b) - y.ravel()).reshape(y.shape)
            s = _scale(r)
            seeds.append((np.mean(np.log(s)), np.r_[fast, slow, b], r))
    seeds.sort(key=lambda p: p[0])
    scale = 2.5 * _scale(seeds[0][2])
    low = [.050001, .050001] + [-np.inf] * (len(seeds[0][1]) - 2)
    high = [24.999, 24.999] + [np.inf] * (len(low) - 2)

    def residual(p):
        # The history plastic mode always tracks the fast branch.
        tau = np.sort(p[:2])
        return ((design.matrix(tau) @ p[2:] - y) / scale).ravel()

    results = [least_squares(residual, p, bounds=(low, high), loss='cauchy',
                             x_scale='jac', max_nfev=160) for _, p, _ in seeds[:4]]
    best = min(results, key=lambda fit: fit.cost)
    scale = 2.5 * _scale(design.matrix(np.sort(best.x[:2])) @ best.x[2:] - y)
    best = least_squares(residual, best.x, bounds=(low, high), loss='cauchy',
                         x_scale='jac', max_nfev=160)
    tau, weights = np.sort(best.x[:2]), best.x[2:]
    return tau, np.pad(weights, (0, 9 - len(weights)))


def solve_alpha_m96_w02_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    calib = _calibrate(dataset['calibration_load_cells'])
    x = _align(train, test)
    n = len(train['lab_id'])
    args = (calib, dataset['past_dt'], dataset['eval_times'], dataset['yield_stress'])
    tr = _Design(train, x[:n], *args)
    target = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                              train['instant_rate_obs']))
    tau, weights = _fit(tr, target)
    te = _Design(test, x[n:], *args)
    a = te.matrix(tau)
    prediction = a @ weights[:a.shape[-1]]
    # Incomplete covariates leave the linear drift uncertain. After removing
    # the fitted constitutive rate, the measured initial rate identifies the
    # missing drift without changing the relaxation or plastic-work terms.
    if 'instant_rate_obs' in test:
        observed = np.asarray(test['instant_rate_obs'])
        incomplete = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
        use = incomplete & np.isfinite(observed)
        innovation = observed[use] - prediction[use, -1]
        prediction[use, :-1] += innovation[:, None] * te.time
        prediction[use, -1] += innovation
    return dict(estimated_taus=tau, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:],
                predicted_strain=prediction[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=prediction[:, -1])
