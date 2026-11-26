"""Calibrate coupon measurements and fit the sub-yield hereditary hold law."""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _mad(x, axis=None):
    center = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - center), axis=axis)


_ARCHIVE = Path('/tmp/hegel_continual_archive')
_CAL_FIELDS = ('lab_id', 'cmd_stress', 'true_stress')


def _checks(c, lab):
    ids, x, y = (np.asarray(c[k]) for k in _CAL_FIELDS)
    good = (ids == lab) & np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _affine(x, y):
    """Find the largest agreeing group of small load-cell check panels."""
    best = np.array([1., np.median(y-x) if len(x) else 0.])
    keep = np.zeros(len(x), dtype=bool)
    score = (0, 0.)
    for i, j in zip(*np.triu_indices(len(x), 1)):
        dx = x[j] - x[i]
        if abs(dx) < .05:
            continue
        gain = (y[j] - y[i]) / dx
        if gain <= 0:
            continue
        bias = y[i] - gain*x[i]
        r = np.abs(y-gain*x-bias)
        use = r <= 8e-4
        trial = (int(use.sum()), np.ptp(x[use]))
        if trial > score:
            score, keep = trial, use
    if keep.sum() >= 2:
        best = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))),
                               y[keep], rcond=None)[0]
    return best, keep.sum() >= 3 and score[1] >= .25


def _archive_calibrations(labs):
    """Read recent archived checks when the current panel lacks leverage."""
    def panels(record):
        if isinstance(record, dict):
            if all(k in record for k in _CAL_FIELDS):
                yield record
            else:
                for key in ('calibration_load_cells', 'dataset', 'batches'):
                    if key in record:
                        yield from panels(record[key])
        elif isinstance(record, (tuple, list)):
            for entry in reversed(record):
                yield from panels(entry)

    found = {}
    files = (f for f in _ARCHIVE.rglob('*') if f.is_file() and
             f.suffix.lower() in ('.json', '.npz', '.pkl', '.pickle'))
    for path in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            if path.suffix.lower() == '.npz':
                with np.load(path, allow_pickle=False) as data:
                    record = {k: data[k] for k in _CAL_FIELDS if k in data}
            elif path.suffix.lower() == '.json':
                with path.open() as f:
                    record = json.load(f)
            else:
                with path.open('rb') as f:
                    record = pickle.load(f)
            for c in panels(record):
                for lab in labs:
                    if lab not in found:
                        fit, reliable = _affine(*_checks(c, lab))
                        if reliable:
                            found[lab] = fit
        except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError):
            continue
        if len(found) == len(labs):
            break
    return found


def _calibrate(c):
    results = [_affine(*_checks(c, lab)) for lab in range(4)]
    cal = np.array([fit for fit, _ in results])
    reliable = np.array([ok for _, ok in results])
    if reliable.all():
        return cal, reliable
    weak = np.flatnonzero(~reliable)
    archived = _archive_calibrations(weak)
    for lab in weak:
        x, y = _checks(c, lab)
        if lab in archived:
            cal[lab] = archived[lab]
        if len(x):
            b = y-cal[lab, 0]*x
            # Isolated spot checks constrain the intercept at the chosen gain.
            if len(b) < 3 or _mad(b) < .01:
                cal[lab, 1] = np.median(b)
    return cal, reliable


def _align(train, test):
    labs, strata, raw, ref = [np.concatenate([s[k] for s in (train, test)]) for k in
                             ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons')]
    x = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        use = labs == lab
        panel = ref[use]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        spread = np.median(np.nanstd(panel, axis=0))
        x[use] = (raw[use] - offset) / spread
    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    w = 1. / counts[cell]
    present = np.isfinite(x)
    support = w @ present
    fill = np.divide(w @ np.where(present, x, 0.), support,
                     out=np.zeros(x.shape[1]), where=support > 0)
    x = np.where(present, x, fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _physical(s, calibration):
    gain, bias = calibration[s['lab_id']].T
    return gain[:, None]*s['past_stress_cmd'] + bias[:, None], gain*s['stress_hold_cmd'] + bias


class _HoldLaw:
    def __init__(self, stress, hold, features, dt, t):
        self.stress, self.hold, self.t = stress, hold, t
        self.end = np.cumsum(dt[::-1])[::-1] - dt
        self.start = self.end + dt
        self.trend = np.column_stack((np.ones(len(hold)), features))
        self.linear = self.trend[:, None, :] * np.r_[t, 1.][None, :, None]

    def design(self, taus, derivative=False):
        n, k = len(self.hold), len(self.t)
        a = np.empty((n, k+1, 7))
        a[:, :, 2:] = self.linear
        da = np.empty((n, k+1, 2)) if derivative else None
        for j, tau in enumerate(taus):
            e0, e1 = np.exp(-self.end/tau), np.exp(-self.start/tau)
            gap = self.hold - self.stress @ (e0 - e1)
            et = np.exp(-self.t/tau)
            f = np.r_[-np.expm1(-self.t/tau), 1./tau]
            a[:, :, j] = gap[:, None]*f
            if derivative:
                dg = -(self.stress @ (self.end*e0 - self.start*e1))/tau**2
                df = np.r_[-self.t*et, -1.]/tau**2
                da[:, :, j] = dg[:, None]*f + gap[:, None]*df
        return a, da


def _estimate(law, y):
    def fit(p, scale, loss):
        def fun(q):
            a, _ = law.design(q[:2])
            return ((a @ q[2:] - y) / scale).ravel()

        def jac(q):
            a, da = law.design(q[:2], derivative=True)
            j = np.concatenate((da*q[2:4], a), axis=2)
            return (j / scale[..., None]).reshape(-1, 9)

        return least_squares(fun, p, jac=jac, x_scale='jac', loss=loss,
                             bounds=(np.r_[.20, .20, np.full(7, -np.inf)],
                                     np.r_[8., 8., np.full(7, np.inf)]), max_nfev=180)

    scale = np.full((1, y.shape[1]), .02)
    seeds = []
    for taus in ((.35, 1.8), (.7, 3.8), (1.4, 6.5), (.25, 6.5)):
        a, _ = law.design(taus)
        theta = np.linalg.lstsq(a.reshape(-1, 7), y.ravel(), rcond=None)[0]
        seeds.append(fit(np.r_[taus, theta], scale, 'soft_l1'))
    best = min(seeds, key=lambda r: r.cost)
    r = best.fun.reshape(y.shape)*scale
    # Strain and the independent onset rate have different observation noise.
    scale[0, :-1] = max(float(_mad(r[:, :-1].ravel())), 1e-6)*2.5
    scale[0, -1] = max(float(_mad(r[:, -1])), 1e-6)*2.5
    p = fit(best.x, scale, 'cauchy').x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p[:2], p[2:]


def _recover_weak_gains(train, c, cal, reliable, features, complete, dt, t, y):
    labs = train['lab_id']
    trusted = complete & reliable[labs]
    if reliable.all() or trusted.sum() < 12:
        return
    h, s = _physical(train, cal)
    tau, theta = _estimate(_HoldLaw(h[trusted], s[trusted], features[trusted], dt, t), y[trusted])
    a, _ = _HoldLaw(h[trusted], s[trusted], features[trusted], dt, t).design(tau)
    r = a @ theta-y[trusted]
    scale = 2.5*np.maximum([_mad(r[:, :-1].ravel()), _mad(r[:, -1])], 1e-6)
    scale = np.r_[np.full(len(t), scale[0]), scale[1]]
    # A common stress offset cancels except for the memory of z(0)=0.
    transient = theta[:2]*np.exp(-dt.sum()/tau)
    bias_shape = np.r_[(-np.expm1(-t[:, None]/tau)) @ transient, transient @ (1./tau)]
    for lab in np.flatnonzero(~reliable):
        m = complete & (labs == lab)
        if m.sum() < 4:
            continue
        law = _HoldLaw(train['past_stress_cmd'][m], train['stress_hold_cmd'][m], features[m], dt, t)
        a, _ = law.design(tau)
        gain_shape = a[..., :2] @ theta[:2]
        target = y[m] - a[..., 2:] @ theta[2:]
        x, checks = _checks(c, lab)
        prior_bias = cal[lab, 1]

        def bias(gain):
            if len(x):
                b = checks - gain*x
                if len(b) < 3 or _mad(b) < .01:
                    return np.median(b)
            return prior_bias

        def residual(p):
            gain = p[0]
            return ((gain*gain_shape + bias(gain)*bias_shape - target)/scale).ravel()

        gain = least_squares(residual, cal[lab, :1], bounds=(1e-5, np.inf),
                             loss='cauchy').x[0]
        cal[lab] = gain, bias(gain)


def solve_alpha_w03_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    checks = dataset['calibration_load_cells']
    cal, reliable = _calibrate(checks)
    xtr, xte = _align(train, test)
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    _recover_weak_gains(train, checks, cal, reliable, xtr, complete, dt, t, y)
    htr, str_ = _physical(train, cal)
    taus, theta = _estimate(_HoldLaw(htr[complete], str_[complete], xtr[complete], dt, t), y[complete])
    a, _ = _HoldLaw(*_physical(test, cal), xte, dt, t).design(taus)
    pred = a @ theta
    # The residual at onset identifies the unresolved linear drift on imputed rows.
    onset = test['instant_rate_obs']
    anchor = ~np.isfinite(test['coupon_features_raw']).all(axis=1) & np.isfinite(onset)
    drift = onset[anchor] - pred[anchor, -1]
    pred[anchor, :-1] += drift[:, None] * t
    pred[anchor, -1] = onset[anchor]
    return dict(estimated_taus=taus, hereditary_weights=theta, viscoplastic_weights=np.zeros(2),
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None], initial_rate_pred=pred[:, -1])
