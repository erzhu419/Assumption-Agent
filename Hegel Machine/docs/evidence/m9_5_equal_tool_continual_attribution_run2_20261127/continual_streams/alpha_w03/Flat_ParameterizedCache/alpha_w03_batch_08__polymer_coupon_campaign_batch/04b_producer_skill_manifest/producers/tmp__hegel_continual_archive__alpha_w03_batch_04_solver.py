"""Calibrated two-mode hereditary creep with quadratic overstress."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _checks(calibration, lab):
    x = np.asarray(calibration['cmd_stress'], dtype=float)
    y = np.asarray(calibration['true_stress'], dtype=float)
    ok = (np.asarray(calibration['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _supported_line(x, y):
    if len(x) < 4 or np.ptp(x) < .5:
        return None
    # Select the line supported by the largest number of checks at sweep precision.
    i, j = np.triu_indices(len(x), 1)
    use = np.abs(x[i]-x[j]) >= .25
    i, j = i[use], j[use]
    g = (y[i]-y[j])/(x[i]-x[j])
    b = y[i]-g*x[i]
    residual = np.abs(y[:, None] - x[:, None]*g - b)
    counts = (residual <= .002).sum(axis=0)
    best = np.lexsort((np.minimum(residual, .002).sum(axis=0), -counts))[0]
    clean = residual[:, best] <= .002
    if clean.sum() < max(4, len(x)//2+1) or np.ptp(x[clean]) < .5:
        return None
    return np.linalg.lstsq(np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]


def _archive_calibration(labs):
    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    records = json.loads(path.read_text()) if path.is_file() else []
    return {lab: [fit for row in reversed(records)
                  if (fit := _supported_line(*_checks(row, lab))) is not None]
            for lab in labs}


def _sparse_line(x, y, priors):
    if not priors:
        return (1., np.median(y-x) if len(x) else 0.)
    if len(x) == 0:
        return priors[0]
    a = np.asarray(priors)
    err = np.abs(y[:, None] - x[:, None]*a[:, 0] - a[:, 1])
    support = err <= .003
    counts = support.sum(axis=0)
    if counts.max():
        best = np.lexsort((np.minimum(err, .003).sum(axis=0), -counts))[0]
        return a[best]
    # Spot checks without a matching historical regime can update the intercept,
    # but cannot identify a new slope.
    best = np.argmin(np.median(err, axis=0))
    g, b = a[best]
    offsets = y-g*x
    middle = np.median(offsets)
    if len(x) >= 2 and np.count_nonzero(np.abs(offsets-middle) < .003) > len(x)/2:
        b = middle
    return g, b


def _prep(d):
    c = d['calibration_load_cells']
    lab = np.concatenate([d[k]['lab_id'] for k in ('train', 'test')])
    strata = np.concatenate([d[k]['stratum'] for k in ('train', 'test')])
    x = np.concatenate([d[k]['coupon_features_raw'] for k in ('train', 'test')]).astype(float)
    ref = np.concatenate([d[k]['ref_coupons'] for k in ('train', 'test')])
    h = np.concatenate([d[k]['stress_hold_cmd'] for k in ('train', 'test')]).astype(float)
    p = np.concatenate([d[k]['past_stress_cmd'] for k in ('train', 'test')]).astype(float)
    checks = {i: _checks(c, i) for i in np.unique(lab)}
    fits = {i: _supported_line(*xy) for i, xy in checks.items()}
    sparse = [i for i in fits if fits[i] is None]
    priors = _archive_calibration(sparse) if sparse else {}
    for i, fit in fits.items():
        m = lab == i
        g, b = fit if fit is not None else _sparse_line(*checks[i], priors[i])
        h[m] = g*h[m] + b
        p[m] = g*p[m] + b
        rc = ref[m]
        usable = np.isfinite(rc).sum(axis=0) >= 2
        rc = rc[:, usable]
        if rc.size:
            center = np.median(np.nanmean(rc, axis=0))
            scale = np.median(np.nanstd(rc, axis=0, ddof=0))
            x[m] = (x[m] - center) / (scale if scale > 0 else 1.)
    # Give each observed lab/stratum cell equal influence, separately per column.
    cells = [x[(lab == i) & (strata == j)] for i, j in np.unique(np.column_stack((lab, strata)), axis=0)]
    sums = np.array([np.nansum(a, axis=0) for a in cells])
    counts = np.array([np.isfinite(a).sum(axis=0) for a in cells])
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    good = (counts > 0).sum(axis=0)
    fill = np.divide(means.sum(axis=0), good, out=np.zeros(x.shape[1]), where=good > 0)
    return np.column_stack((np.ones(len(h)), np.where(np.isfinite(x), x, fill))), h, p


def _design(tau, x, h, p, dt, t, yield_stress):
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    decay = np.exp(-after[:, None]/tau)
    jump = -np.expm1(-dt[:, None]/tau)
    gap = h[:, None] - p @ (decay*jump)
    shape = -np.expm1(-t[:, None]/tau)
    ov = np.maximum(np.abs(h)-yield_stress, 0.)**2 * np.sign(h)
    work = (np.maximum(np.abs(p)-yield_stress, 0.)**2 @ dt) * h
    curves = np.concatenate((gap[:, None, :]*shape,
                             x[:, None, :]*t[None, :, None],
                             ov[:, None, None]*t[None, :, None],
                             work[:, None, None]*shape[None, :, :1]), axis=2)
    rates = np.column_stack((gap/tau, x, ov, work/tau[0]))
    return curves, rates


def _fit(x, h, p, dt, t, threshold, y, rate):
    yc, yr = np.isfinite(y), np.isfinite(rate)
    obs = np.r_[y[yc], rate[yr]]
    rate_start = yc.sum()

    def decode(v):
        return np.array([v[0], v[0]+(24.99-v[0])*v[1]])

    def design(v):
        curves, rates = _design(decode(v), x, h, p, dt, t, threshold)
        return np.vstack((curves[yc], rates[yr]))

    def profile(v, w):
        a = design(v)
        coef = np.linalg.lstsq(a*w[:, None], obs*w, rcond=1e-10)[0]
        return a @ coef - obs, coef

    def opt(v, w):
        return least_squares(lambda q: w*profile(q, w)[0], v,
                             bounds=([.0501, 1e-5], [24., .99999]),
                             ftol=1e-10, max_nfev=80)

    def spread(r):
        return max(.0001, 1.4826*np.median(np.abs(r-np.median(r)))) if r.size else 1.

    w = np.ones(len(obs))
    starts = [(a, (b-a)/(24.99-a)) for a, b in ((.2, 1.4), (.5, 2.8), (1.2, 5.), (2.5, 10.))]
    v = min((opt(start, w) for start in starts), key=lambda r: r.cost).x
    for i in range(6):
        r, coef = profile(v, w)
        sc = spread(r[:rate_start]); sr = spread(r[rate_start:])
        base = np.r_[np.full(rate_start, sc), np.full(len(obs)-rate_start, sr)]
        w = (1./base) / np.hypot(1., r/(3.*base))
        w /= np.max(w)
        v = opt(v, w).x
    theta = profile(v, w)[1]
    if not np.any(np.abs(h) > threshold):
        theta[7] = 0.
    if not np.any(np.abs(p) > threshold):
        theta[8] = 0.
    return decode(v), theta


def solve_alpha_w03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    threshold = dataset['yield_stress']
    x, h, p = _prep(dataset)
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    # An imputed feature is not a measurement of a coupon's drift. Use
    # complete coupons for fitting when they span the active plastic terms.
    over_hold = np.abs(h[:n]) > threshold
    over_hist = np.any(np.abs(p[:n]) > threshold, axis=1) & (h[:n] != 0.)
    if (complete.sum() >= 9 and
            (not over_hold.any() or over_hold[complete].any()) and
            (not over_hist.any() or over_hist[complete].any())):
        fit = np.flatnonzero(complete)
    else:
        fit = np.arange(n)
    tau, theta = _fit(x[fit], h[fit], p[fit], dt, t, threshold,
                      (train['strain_curves_obs'] - train['strain_0'][:, None])[fit],
                      train['instant_rate_obs'][fit])
    curves, rates = _design(tau, x[n:], h[n:], p[n:], dt, t, threshold)
    strain = test['strain_0'][:, None] + curves @ theta
    rate = rates @ theta
    measured = np.asarray(test['instant_rate_obs'])
    anchor = ~np.isfinite(test['coupon_features_raw']).all(axis=1) & np.isfinite(measured)
    # Features enter only the linear drift, so its missing contribution has
    # the same slope at onset and throughout the hold.
    strain[anchor] += (measured[anchor] - rate[anchor])[:, None] * t
    rate[anchor] = measured[anchor]
    return dict(estimated_taus=tau, hereditary_weights=theta[:7],
                viscoplastic_weights=theta[7:], predicted_strain=strain,
                initial_rate_pred=rate)
