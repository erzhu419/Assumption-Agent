"""Affine-calibrated, two-branch sub-yield hereditary creep."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares


def _line(x, y):
    """Resistant pairwise slope start, then refit with bounded influence."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[i] - x[j]
    good = np.abs(dx) > 1e-6
    if not good.any():
        return np.array([1., np.median(y - x) if len(x) else 0.])
    slope = np.median((y[i[good]] - y[j[good]]) / dx[good])
    b = np.array([slope, np.median(y - slope*x)])
    a = np.column_stack((x, np.ones(len(x))))
    for _ in range(8):
        r = y - a @ b
        s = max(1e-4, 1.4826*np.median(np.abs(r - np.median(r))))
        w = 1./np.hypot(1., r/(2.5*s))
        b = np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]
    return b


_ARCHIVE = Path('/tmp/hegel_continual_archive')


def _checks(calibration, lab):
    x = np.asarray(calibration['cmd_stress'], dtype=float)
    y = np.asarray(calibration['true_stress'], dtype=float)
    use = (np.asarray(calibration['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _supported_line(x, y):
    if len(x) < 4 or np.ptp(x) < 0.5:
        return None
    fit = _line(x, y)
    residual = y - (fit[0]*x + fit[1])
    scatter = 1.4826*np.median(np.abs(residual - np.median(residual)))
    clean = np.abs(residual) <= max(0.002, 4.*min(scatter, 0.005))
    if clean.sum() < max(4, len(x)//2 + 1) or np.ptp(x[clean]) < 0.5:
        return None
    return fit if clean.all() else _line(x[clean], y[clean])


def _historical_lines():
    path = _ARCHIVE / 'archive_calibration_history.json'
    if not path.exists():
        return []
    with path.open() as stream:
        return json.load(stream)


def _sparse_line(x, y, archive, lab):
    priors = [_supported_line(*_checks(table, lab)) for table in reversed(archive)]
    priors = [p for p in priors if p is not None] or [np.array([1., 0.])]
    if not len(x):
        return priors[0]
    # A separated current pair identifies both affine parameters, even after recalibration.
    if len(x) == 2 and np.ptp(x) >= .5:
        return _line(x, y)
    # A third point can independently confirm a shifted line.
    if len(x) == 3 and np.ptp(x) >= .5:
        trial = _line(x, y)
        if np.max(np.abs(y - (trial[0]*x + trial[1]))) <= .002:
            return trial
    # Otherwise use the archive to identify compatible spots or supply the missing slope.
    def score(p):
        r = np.abs(y - p[0]*x - p[1])
        return (-np.count_nonzero(r < .02), np.median(r), np.min(r))
    gain, bias = min(priors, key=score)
    delta = y - (gain*x + bias)
    close = np.abs(delta) < .02
    if close.sum() >= 2 and np.ptp(x[close]) >= .5:
        return _line(x[close], y[close])
    if close.any():
        return gain, bias + np.median(delta[close])
    if len(delta) >= 2 and np.ptp(delta) <= .004:
        return gain, bias + np.median(delta)
    return gain, bias


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
    archive = _historical_lines() if sparse else []
    for i, fit in fits.items():
        m = lab == i
        g, b = fit if fit is not None else _sparse_line(*checks[i], archive, i)
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


def _design(tau, x, h, p, memory, dt, t):
    tau = np.asarray(tau)
    # Unrolling the recurrence gives the exact weights of the past stress steps.
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a = after[:, None]/tau
    b = dt[:, None]/tau
    decay = np.exp(-a)
    jump = -np.expm1(-b)
    carry = memory * np.exp(-dt.sum()/tau)
    gap = h[:, None] - p @ (decay*jump) - carry
    dgap = -(p @ (decay*(a*jump - b*np.exp(-b))/tau))
    dgap -= carry * dt.sum()/tau**2
    q = t[:, None]/tau
    shape = -np.expm1(-q)
    dshape = -np.exp(-q)*q/tau
    curves = np.concatenate((gap[:, None, :]*shape, x[:, None, :]*t[None, :, None]), axis=2)
    rates = np.column_stack((gap/tau, x))
    dc = dgap[:, None, :]*shape + gap[:, None, :]*dshape
    dr = (dgap - gap/tau)/tau
    return curves, rates, dc, dr


def _fit(x, h, p, memory, dt, t, y, rate):
    n, k = y.shape
    obs = np.r_[y.ravel(), rate]
    def design(tau):
        c, r, dc, dr = _design(tau, x, h, p, memory, dt, t)
        return np.vstack((c.reshape(-1, 7), r)), np.vstack((dc.reshape(-1, 2), dr))

    # Nested coordinates enforce the box and the branch ordering throughout.
    def decode(v):
        return np.array([v[0], v[0] + (8.-v[0])*v[1]])

    def optimize(v, scales, loss):
        cache = [None, None, None]
        def parts(u):
            if cache[0] is None or not np.array_equal(cache[0], u[:2]):
                cache[:] = [u[:2].copy(), *design(decode(u))]
            return cache[1:]
        def fun(u):
            a, _ = parts(u)
            return (a @ u[2:] - obs)/scales
        def jac(u):
            a, d = parts(u)
            d = d*u[2:4]
            j = np.column_stack((d[:, 0] + (1.-u[1])*d[:, 1], (8.-u[0])*d[:, 1], a))
            return j/scales[:, None]
        return least_squares(fun, v, jac=jac, loss=loss, x_scale='jac',
                             bounds=([.2, 1e-6] + [-np.inf]*7, [7.9999, 1.] + [np.inf]*7),
                             max_nfev=150, ftol=2e-10)

    def spread(r):
        return max(1e-6, 1.4826*np.median(np.abs(r - np.median(r))))

    def scales_for(res, factor):
        return np.r_[np.full(n*k, factor*spread(res[:n*k])), np.full(n, factor*spread(res[n*k:]))]

    starts = []
    for lo, hi in ((.24, 1.4), (.4, 2.8), (.75, 4.0), (1.4, 6.5), (2.9, 7.6)):
        a, _ = design([lo, hi])
        coef = np.linalg.lstsq(a, obs, rcond=None)[0]
        ini = np.r_[lo, (hi-lo)/(8.-lo), coef]
        starts.append(ini)
    pilot = starts[1]
    a, _ = design(decode(pilot))
    scales = scales_for(a @ pilot[2:] - obs, 2.5)
    coarse = min((optimize(v, scales, 'soft_l1') for v in starts), key=lambda r: r.cost)
    a, _ = design(decode(coarse.x))
    scales = scales_for(a @ coarse.x[2:] - obs, 2.8)
    result = optimize(coarse.x, scales, 'cauchy')
    return decode(result.x), result.x[2:]


def _crossing(tau, amplitude, drift, threshold, end):
    """First upward crossing of the continuous two-mode trajectory."""
    if threshold <= 0.:
        return 0.
    inv = 1./tau
    rates = amplitude * inv

    def level(t):
        return -np.expm1(-t*inv) @ amplitude + drift*t - threshold

    def slope(t):
        return np.exp(-t*inv) @ rates + drift

    # The curvature has at most one zero, so slope is monotone on each interval.
    curvature = rates * inv
    breaks = [0., end]
    if np.sign(curvature[0]) * np.sign(curvature[1]) < 0:
        pivot = (np.log(abs(curvature[0])) - np.log(abs(curvature[1]))) / (inv[0]-inv[1])
        if 0. < pivot < end:
            breaks.insert(1, pivot)
    extrema = [0.]
    for a, b in zip(breaks[:-1], breaks[1:]):
        sa, sb = slope(a), slope(b)
        if sa*sb < 0.:
            extrema.append(brentq(slope, a, b))
        elif sb == 0. and b < end:
            extrema.append(b)
    for a, b in zip(extrema, extrema[1:] + [end]):
        if level(b) >= 0.:
            return brentq(level, a, b)
    return end


def solve_alpha_w03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x, h, p = _prep(dataset)
    memory = np.concatenate([train['initial_memory_states'], test['initial_memory_states']])
    tau, theta = _fit(x[:n], h[:n], p[:n], memory[:n], dt, t,
                      train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    curves, rates, _, _ = _design(tau, x[n:], h[n:], p[n:], memory[n:], dt, t)
    amplitudes = rates[:, :2]*tau*theta[:2]
    drift = x[n:] @ theta[2:]
    # An initial-rate observation recovers the coupon-specific contribution
    # lost to mean imputation, without perturbing the two memory amplitudes.
    observed_rate = np.asarray(test['instant_rate_obs'])
    incomplete = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    use_rate = incomplete & np.isfinite(observed_rate)
    drift[use_rate] = observed_rate[use_rate] - rates[use_rate, :2] @ theta[:2]
    inc = curves[:, :, :2] @ theta[:2] + drift[:, None]*t
    threshold = test['critical_strain'] - test['strain_0']
    cross = [_crossing(tau, a, v, q, t[-1]) for a, v, q in zip(amplitudes, drift, threshold)]
    return dict(estimated_taus=tau, predicted_strain=test['strain_0'][:, None] + inc,
                secant_compliance=inc[:, -1]/np.maximum(np.abs(h[n:]), .5),
                hazard_crossing_times=np.asarray(cross))
