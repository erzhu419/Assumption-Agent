import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _rows(table, lab):
    x, y = (np.asarray(table[k], float) for k in ('cmd_stress', 'true_stress'))
    ok = (np.asarray(table['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _line(x, y, tol, span):
    i, j = np.triu_indices(len(x), 1)
    ok = abs(x[j] - x[i]) >= span
    i, j = i[ok], j[ok]
    if not len(i):
        return None
    gain = (y[j] - y[i]) / (x[j] - x[i])
    r = abs(y - y[i, None] - gain[:, None] * (x - x[i, None]))
    inlier = r <= tol
    count = inlier.sum(axis=1)
    best = np.lexsort(((r * inlier).sum(axis=1), -count))[0]
    if count[best] < max(3, len(x) // 2 + 1):
        return None
    use = inlier[best]
    return np.linalg.lstsq(np.c_[x[use], np.ones(use.sum())], y[use], rcond=None)[0]


def _calibrate(checks, scale):
    coeff = []
    history = None
    tol, span = 0.0005 * scale, 0.1 * scale
    for lab in range(4):
        x, y = _rows(checks, lab)
        fit = _line(x, y, tol, span)
        if fit is None:
            if history is None:
                history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.exists() else []
            prior = [_line(*_rows(table, lab), tol, span) for table in history]
            prior = [line for line in prior if line is not None]
            fit = np.median(prior, axis=0) if prior else np.array([1., 0.])
            offset = y - fit[0] * x - fit[1]
            if len(x):
                # Archive slope for a narrow panel; update offset only from coherent checks.
                near = abs(offset) <= tol
                if np.any(near):
                    fit[1] += np.median(offset[near])
        coeff.append(fit)
    return np.asarray(coeff)


def _features(train, test):
    lab, stratum = (np.r_[train[k], test[k]] for k in ('lab_id', 'stratum'))
    feat = np.vstack([s['coupon_features_raw'] for s in (train, test)]).astype(float)
    ref = np.vstack([s['ref_coupons'] for s in (train, test)])
    for c in np.unique(lab):
        rows = lab == c
        r = ref[rows]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        feat[rows] = (feat[rows] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    _, cell = np.unique(np.c_[lab, stratum], axis=0, return_inverse=True)
    cell = cell.ravel()
    means = []
    for k in np.unique(cell):
        a = feat[cell == k]
        n = np.isfinite(a).sum(axis=0)
        means.append(np.divide(np.nansum(a, axis=0), n, out=np.full(a.shape[1], np.nan), where=n > 0))
    means = np.asarray(means)
    count = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), count, out=np.zeros(feat.shape[1]), where=count > 0)
    return np.c_[np.ones(len(feat)), np.where(np.isfinite(feat), feat, fill)]


def _stress(split, coeff):
    gain, bias = coeff[split['lab_id']].T
    return (split['past_stress_cmd'] * gain[:, None] + bias[:, None],
            split['stress_hold_cmd'] * gain + bias, split['initial_memory_states'])


def _memory(stress, dt, tau, jac=False):
    past, hold, initial = stress
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a, b = remaining[:, None] / tau, dt[:, None] / tau
    ea = np.exp(-a)
    mix = ea * -np.expm1(-b)
    total = dt.sum() / tau
    seed = initial * np.exp(-total)
    delta = hold[:, None] - past @ mix - seed
    if jac:
        return delta, -(past @ (a * mix - ea * b * np.exp(-b))) - seed * total
    return delta


def _design(stress, feat, dt, t, logtau, jac=False):
    tau = np.exp(logtau)
    d = _memory(stress, dt, tau, jac)
    if jac:
        d, dd = d
    u = t[:, None] / tau
    f = -np.expm1(-u)
    curve = np.concatenate((d[:, None, :] * f, feat[:, None, :] * t[None, :, None]), axis=2)
    A = np.concatenate((curve, np.c_[d / tau, feat][:, None, :]), axis=1)
    if not jac:
        return A
    dcurve = dd[:, None, :] * f - d[:, None, :] * (u * np.exp(-u))
    return A, np.concatenate((dcurve, ((dd - d) / tau)[:, None, :]), axis=1)


def _fit(stress, feat, dt, t, y):
    keep = np.isfinite(y)
    obs = y[keep]
    s = np.ones(len(obs))

    def residual(p):
        return (_design(stress, feat, dt, t, p[:2])[keep] @ p[2:] - obs) / s

    def jac(p):
        a, da = _design(stress, feat, dt, t, p[:2], True)
        return np.c_[da[keep] * p[None, 2:4], a[keep]] / s[:, None]

    bounds = (np.r_[np.log([.08, .08]), np.full(7, -np.inf)],
              np.r_[np.log([15., 15.]), np.full(7, np.inf)])
    starts = []
    grid = np.geomspace(.18, 9., 8)
    for i, a in enumerate(grid[:-1]):
        for b in grid[i + 1:]:
            logtau = np.log([a, b])
            A = _design(stress, feat, dt, t, logtau)[keep]
            coef = np.linalg.lstsq(A, obs, rcond=None)[0]
            for _ in range(4):
                w = 1 / np.hypot(1, (A @ coef - obs) / .04)
                coef = np.linalg.lstsq(A * w[:, None], obs * w, rcond=None)[0]
            starts.append((np.log1p(((A @ coef - obs) / .04) ** 2).sum(), np.r_[logtau, coef]))

    def fit(start, scale):
        return least_squares(residual, start, jac=jac, bounds=bounds, loss='cauchy',
                             f_scale=scale, x_scale='jac', max_nfev=120)

    first = [fit(p, .04) for _, p in sorted(starts, key=lambda a: a[0])[:3]]
    p = min(first, key=lambda v: v.cost).x
    errors = _design(stress, feat, dt, t, p[:2]) @ p[2:] - y
    scales = [max(1e-5, 1.4826 * np.nanmedian(abs(a - np.nanmedian(a))))
              for a in (errors[:, :-1], errors[:, -1])]
    s = np.broadcast_to(np.r_[np.repeat(scales[0], len(t)), scales[1]], y.shape)[keep]
    p = fit(p, 2.).x
    return np.exp(p[:2]), p[2:]


def _crossings(times, strain, start, critical):
    t = np.r_[0., times]
    z = np.c_[start, strain]
    above = z >= critical[:, None]
    out = np.full(len(z), t[-1], dtype=float)
    out[above[:, 0]] = 0.
    k = above.argmax(axis=1)
    rows = np.flatnonzero(k > 0)
    right = k[rows]
    left = right - 1
    fraction = (critical[rows] - z[rows, left]) / (z[rows, right] - z[rows, left])
    out[rows] = t[left] + (t[right] - t[left]) * fraction
    return out


def solve_alpha_w02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    coeff = _calibrate(dataset['calibration_load_cells'], dataset['yield_stress'])
    X = _features(train, test)
    n = len(train['lab_id'])
    y = np.c_[train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']]
    tau, p = _fit(_stress(train, coeff), X[:n], dt, t, y)
    stress = _stress(test, coeff)
    amplitudes = _memory(stress, dt, tau) * p[:2]
    drift = X[n:] @ p[2:]
    # The initial rate observes the missing feature's contribution to drift.
    rate_drift = test['instant_rate_obs'] - (amplitudes / tau).sum(axis=1)
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    drift = np.where(missing & np.isfinite(rate_drift), rate_drift, drift)
    strain0 = test['strain_0']
    delta = amplitudes @ -np.expm1(-t[None, :] / tau[:, None]) + drift[:, None] * t
    pred = strain0[:, None] + delta
    crossing = _crossings(t, pred, strain0, test['critical_strain'])
    return {'estimated_taus': np.sort(tau), 'predicted_strain': pred,
            'secant_compliance': (pred[:, -1] - strain0) / np.maximum(np.abs(stress[1]), .5),
            'hazard_crossing_times': crossing}
