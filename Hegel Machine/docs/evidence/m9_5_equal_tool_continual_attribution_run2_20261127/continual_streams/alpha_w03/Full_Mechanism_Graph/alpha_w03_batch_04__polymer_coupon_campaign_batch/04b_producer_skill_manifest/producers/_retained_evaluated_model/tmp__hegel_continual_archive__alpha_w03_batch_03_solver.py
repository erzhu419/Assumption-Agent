"""Affine-calibrated, two-branch sub-yield hereditary creep."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


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


def _archive_calibration(labs):
    """Latest supported same-lab line in the historical calibration sweeps."""
    path = _ARCHIVE / 'archive_calibration_history.json'
    if not path.exists():
        return {}
    priors = {}
    with path.open() as stream:
        for table in json.load(stream):
            for lab in labs:
                fit = _supported_line(*_checks(table, lab))
                if fit is not None:
                    priors[lab] = fit
    return priors


def _sparse_line(x, y, historical):
    # An earlier sweep is evidence for a candidate line, not for persistence
    # across batches. A current spot check can choose between that line and
    # nominal calibration; a wide clean pair can further update the gain.
    nominal = (1., 0.)
    priors = (nominal,) if historical is None else (nominal, historical)
    if not len(x):
        return nominal
    residuals = [y - (gain*x + bias) for gain, bias in priors]
    tol = 0.02
    k = min(range(len(priors)), key=lambda k: np.minimum(np.abs(residuals[k]), tol).sum())
    gain, bias = priors[k]
    delta = residuals[k]
    close = np.abs(delta) < tol
    if close.sum() >= 2 and np.ptp(x[close]) >= 0.5:
        return _line(x[close], y[close])
    if close.any():
        return gain, bias + np.median(delta[close])
    if len(x) >= 2 and np.ptp(x) >= 0.5:
        # Permit a new calibration when the current checks form a coherent
        # line, even if neither old nor nominal parameters explain it.
        fit = _line(x, y)
        if np.max(np.abs(y - (fit[0]*x + fit[1]))) <= 0.002:
            return fit
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
    priors = _archive_calibration(sparse) if sparse else {}
    for i, fit in fits.items():
        m = lab == i
        g, b = fit if fit is not None else _sparse_line(*checks[i], priors.get(i))
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


def _design(tau, x, h, p, dt, t, initial):
    tau = np.asarray(tau, dtype=float)
    # Unrolling the recurrence gives the exact weights of the past stress steps.
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a = after[:, None]/tau
    b = dt[:, None]/tau
    decay = np.exp(-a)
    jump = -np.expm1(-b)
    gap = h[:, None] - p @ (decay*jump)
    dgap = -(p @ (decay*(a*jump - b*np.exp(-b))/tau))
    preload = initial*np.exp(-dt.sum()/tau)
    gap -= preload
    dgap -= preload*dt.sum()/tau**2
    q = t[:, None]/tau
    shape = -np.expm1(-q)
    dshape = -np.exp(-q)*q/tau
    curves = np.concatenate((gap[:, None, :]*shape, x[:, None, :]*t[None, :, None]), axis=2)
    rates = np.column_stack((gap/tau, x))
    dc = dgap[:, None, :]*shape + gap[:, None, :]*dshape
    dr = (dgap - gap/tau)/tau
    return curves, rates, dc, dr


def _fit(x, h, p, dt, t, initial, y, rate):
    n, k = y.shape
    obs = np.r_[y.ravel(), rate]
    def design(tau):
        c, r, dc, dr = _design(tau, x, h, p, dt, t, initial)
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


def _crossings(strain0, curves, critical, times):
    """First upward crossing of the piecewise-linear hold trace, censored at its end."""
    t = np.r_[0., times]
    y = np.column_stack((strain0, curves))
    hit = y >= critical[:, None]
    row = np.flatnonzero(hit.any(axis=1))
    idx = hit[row].argmax(axis=1)
    out = np.full(len(y), t[-1])
    out[row[idx == 0]] = 0.
    row, idx = row[idx > 0], idx[idx > 0]
    lower, upper = y[row, idx-1], y[row, idx]
    out[row] = t[idx-1] + (t[idx]-t[idx-1])*(critical[row]-lower)/(upper-lower)
    return out


def solve_alpha_w03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x, h, p = _prep(dataset)
    tau, theta = _fit(x[:n], h[:n], p[:n], dt, t, train['initial_memory_states'],
                      train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    curves, rates, _, _ = _design(tau, x[n:], h[n:], p[n:], dt, t, test['initial_memory_states'])
    increment = curves @ theta
    # The hold derivative at zero is the hereditary rate plus constant drift.
    # When a coupon covariate was imputed, use the independently measured
    # initial rate to correct that coupon's drift, not its memory amplitudes.
    missing = np.isnan(test['coupon_features_raw']).any(axis=1)
    rate = np.asarray(test['instant_rate_obs'])
    use = missing & np.isfinite(rate)
    increment[use] += (rate[use] - rates[use] @ theta)[:, None] * t
    strain0 = np.asarray(test['strain_0'])
    predicted = strain0[:, None] + increment
    return dict(estimated_taus=tau, predicted_strain=predicted,
                secant_compliance=(predicted[:, -1]-strain0)/np.maximum(np.abs(h[n:]), .5),
                hazard_crossing_times=_crossings(strain0, predicted, test['critical_strain'], t))
