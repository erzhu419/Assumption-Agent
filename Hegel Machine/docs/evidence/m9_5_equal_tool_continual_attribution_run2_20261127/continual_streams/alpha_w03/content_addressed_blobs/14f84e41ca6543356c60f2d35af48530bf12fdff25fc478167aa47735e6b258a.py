"""Calibrated two-branch Prony creep with thresholded viscoplasticity."""
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


def _archive_calibration(checks, fits):
    """Select archived lines by current agreement, borrowing evidence within a wing."""
    path = _ARCHIVE / 'archive_calibration_history.json'
    nominal = np.array([1., 0.])
    sweeps = [{lab: nominal for lab in checks}]
    if path.exists():
        with path.open() as stream:
            for table in json.load(stream):
                sweep = {lab: _supported_line(*_checks(table, lab)) for lab in checks}
                sweeps.append({lab: fit for lab, fit in sweep.items() if fit is not None})

    def disagreement(lab, line):
        if fits[lab] is not None:
            # A supported current slope is more informative than a narrow check.
            dg, db = fits[lab] - line
            residual = np.abs(np.array([db - dg, db + dg]))
        else:
            x, y = checks[lab]
            residual = np.abs(y - (line[0]*x + line[1]))
        return np.minimum(residual, .02).mean() if len(residual) else 0.

    priors = {}
    for lab in checks:
        if fits[lab] is not None:
            continue
        peer = lab ^ 1
        x, y = checks[lab]
        candidates = []
        for sweep in sweeps:
            if lab not in sweep:
                continue
            line = sweep[lab]
            residual = np.abs(y - (line[0]*x + line[1]))
            support = np.count_nonzero(residual <= .002)
            peer_error = disagreement(peer, sweep[peer]) if peer in sweep else .02
            candidates.append((line, support, peer_error, disagreement(lab, line)))
        # Two matching checks can identify a local regime. A lone matching
        # check cannot overrule a clean, independently fitted companion lab.
        local_support = max(v[1] for v in candidates)
        if local_support >= 2:
            pool = [v for v in candidates if v[1] == local_support]
            chosen = min(pool, key=lambda v: (v[2], v[3]))
        else:
            peer_best = min(v[2] for v in candidates)
            if fits[peer] is not None and peer_best <= .002:
                pool = [v for v in candidates if v[2] <= peer_best + .001]
                chosen = min(pool, key=lambda v: (-v[1], v[3], v[2]))
            else:
                local_best = min(v[3] for v in candidates)
                pool = [v for v in candidates if v[3] <= local_best + .001]
                chosen = min(pool, key=lambda v: (-v[1], v[2], v[3]))
        priors[lab] = chosen[0], fits[peer] is not None and chosen[2] <= .002
    return priors


def _sparse_line(x, y, prior):
    (gain, bias), corroborated = prior
    delta = y - (gain*x + bias)
    close = np.abs(delta) <= .002
    if close.sum() >= 2 and np.ptp(x[close]) >= .5:
        return _line(x[close], y[close])
    if close.any():
        return gain, bias + np.median(delta[close])
    if not corroborated and len(x) >= 2 and np.ptp(x) >= .5:
        fit = _line(x, y)
        if np.max(np.abs(y - (fit[0]*x + fit[1]))) <= .002:
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
    checks = {i: _checks(c, i) for i in range(4)}
    fits = {i: _supported_line(*xy) for i, xy in checks.items()}
    priors = _archive_calibration(checks, fits) if any(v is None for v in fits.values()) else {}
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


def _overstress(h, p, dt, yield_stress):
    hold = np.maximum(np.abs(h) - yield_stress, 0.)**2 * np.sign(h)
    work = np.maximum(np.abs(p) - yield_stress, 0.)**2 @ dt
    return hold, work * h


def _design(tau, x, h, p, dt, t, hold, work):
    tau = np.asarray(tau)
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a, b = after[:, None] / tau, dt[:, None] / tau
    decay, jump = np.exp(-a), -np.expm1(-b)
    gap = h[:, None] - p @ (decay * jump)
    dgap = -(p @ (decay * (a * jump - b * np.exp(-b)) / tau))
    q = t[:, None] / tau
    shape = -np.expm1(-q)
    dshape = -np.exp(-q) * q / tau
    c = np.concatenate((gap[:, None, :] * shape,
                        x[:, None, :] * t[None, :, None],
                        (hold[:, None] * t)[..., None],
                        (work[:, None] * shape[:, 0])[..., None]), axis=2)
    r = np.column_stack((gap / tau, x, hold, work / tau[0]))
    dc = dgap[:, None, :] * shape + gap[:, None, :] * dshape
    dr = (dgap - gap / tau) / tau
    dw = work[:, None] * dshape[:, 0]
    return c, r, dc, dr, dw, -work / tau[0]**2


def _fit(x, h, p, dt, t, hold, work, y, rate):
    n, k = y.shape
    obs = np.r_[y.ravel(), rate]
    valid = np.isfinite(obs)
    obs = obs[valid]
    active = np.r_[np.ones(7, dtype=bool), np.any(hold), np.any(work)]

    def design(tau):
        c, r, dc, dr, dw, wr = _design(tau, x, h, p, dt, t, hold, work)
        a = np.vstack((c.reshape(-1, 9), r))[valid][:, active]
        d = np.vstack((dc.reshape(-1, 2), dr))[valid]
        return a, d, np.r_[dw.ravel(), wr][valid]

    def decode(v):
        return np.array([v[0], v[0] + (24.99 - v[0]) * v[1]])

    def optimize(v, scales, loss):
        cache = [None, None]

        def parts(u):
            if cache[0] is None or not np.array_equal(cache[0], u[:2]):
                cache[:] = [u[:2].copy(), design(decode(u))]
            return cache[1]

        def fun(u):
            a, _, _ = parts(u)
            return (a @ u[2:] - obs) / scales

        def jac(u):
            a, d, w = parts(u)
            j = d * u[2:4]
            if active[-1]:
                j[:, 0] += w * u[-1]
            return np.column_stack((j[:, 0] + (1. - u[1]) * j[:, 1],
                                    (24.99 - u[0]) * j[:, 1], a)) / scales[:, None]

        return least_squares(fun, v, jac=jac, loss=loss, x_scale='jac',
                             bounds=([.051, 1e-6] + [-np.inf] * active.sum(),
                                     [24.98, 1.] + [np.inf] * active.sum()),
                             max_nfev=150, ftol=1e-10)

    is_rate = np.r_[np.zeros(n*k, dtype=bool), np.ones(n, dtype=bool)][valid]

    def scales_for(res, factor):
        def mad(z):
            return max(1e-6, 1.4826 * np.median(np.abs(z - np.median(z)))) if len(z) else 1.
        return factor * np.where(is_rate, mad(res[is_rate]), mad(res[~is_rate]))

    starts = []
    for lo, hi in ((.24, 1.4), (.45, 2.7), (.9, 4.5), (1.6, 7.), (3., 13.)):
        a, _, _ = design([lo, hi])
        coef = np.linalg.lstsq(a, obs, rcond=None)[0]
        starts.append((np.r_[lo, (hi-lo)/(24.99-lo), coef], a @ coef - obs))
    pilot = min(starts, key=lambda s: np.median(np.abs(s[1][~is_rate])))
    scales = scales_for(pilot[1], 2.5)
    coarse = min((optimize(v, scales, 'soft_l1') for v, _ in starts), key=lambda r: r.cost)
    a, _, _ = design(decode(coarse.x))
    scales = scales_for(a @ coarse.x[2:] - obs, 2.8)
    result = optimize(coarse.x, scales, 'cauchy')
    weights = np.zeros(9)
    weights[active] = result.x[2:]
    return decode(result.x), weights


def solve_alpha_w03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x, h, p = _prep(dataset)
    hold, work = _overstress(h, p, dt, dataset['yield_stress'])
    tau, weights = _fit(x[:n], h[:n], p[:n], dt, t, hold[:n], work[:n],
                        train['strain_curves_obs'] - train['strain_0'][:, None],
                        train['instant_rate_obs'])
    curves, rates, *_ = _design(tau, x[n:], h[n:], p[n:], dt, t, hold[n:], work[n:])
    increment = curves @ weights
    initial_rate = rates @ weights
    # Missing covariates affect only the constant creep drift. Its observed
    # initial-rate innovation can recover their combined contribution without
    # changing either hereditary memory or the fitted material coefficients.
    measured = test.get('instant_rate_obs')
    if measured is not None:
        use = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(measured)
        increment[use] += (measured[use] - initial_rate[use])[:, None] * t
        initial_rate[use] = measured[use]
    return dict(estimated_taus=tau, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:],
                predicted_strain=test['strain_0'][:, None] + increment,
                initial_rate_pred=initial_rate)
