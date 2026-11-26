"""Lab calibration and robust identification of two-branch hereditary creep."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
_CAL_TOL = 1e-3


def _checks(table, lab):
    labs, x, y = (np.asarray(table[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    good = (labs == lab) & np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _line(x, y, prior=None):
    i, j = np.triu_indices(len(x), 1)
    use = abs(x[i] - x[j]) > 1e-6
    i, j = i[use], j[use]
    if not i.size:
        return None, 0, 0.
    g = (y[i] - y[j]) / (x[i] - x[j])
    b = y[i] - g*x[i]
    resid = abs(y - g[:, None]*x - b[:, None])
    masks = resid <= _CAL_TOL
    counts = masks.sum(axis=1)
    choices = np.flatnonzero(counts == counts.max())
    if prior is not None:
        k = min(choices, key=lambda q: (abs(g[q]-prior[0])+abs(b[q]-prior[1]), np.sum(resid[q, masks[q]])))
    else:
        k = min(choices, key=lambda q: (-np.ptp(x[masks[q]]), np.sum(resid[q, masks[q]])))
    good = masks[k]
    return np.linalg.lstsq(np.c_[x[good], np.ones(good.sum())], y[good], rcond=None)[0], int(good.sum()), np.ptp(x[good])


def _calibrate(table):
    curves = []
    archive = None
    for lab in range(4):
        x, y = _checks(table, lab)
        fit, n, span = _line(x, y)
        if n >= 4 and n > len(x)/2 and span >= .5:
            curves.append(fit)
            continue
        if archive is None:
            archive = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.exists() else []
        prior = None
        recent = []
        for sweep in reversed(archive):
            sx, sy = _checks(sweep, lab)
            old, count, width = _line(sx, sy)
            if count < 4 or width < .5:
                continue
            if prior is None:
                prior = old
            elif np.max(np.abs(sy - prior[0]*sx - prior[1])) > 3*_CAL_TOL:
                break
            recent.append((sx, sy))
        if recent:
            px = np.concatenate([u for u, _ in recent])
            py = np.concatenate([v for _, v in recent])
            prior = _line(px, py, prior)[0]
        if prior is None:
            curves.append(fit if fit is not None else (1., np.median(y-x) if len(x) else 0.))
            continue
        fit, n, span = _line(x, y, prior)
        if n >= 3 and n > len(x)/2 and span >= .5:
            curves.append(fit)
            continue
        delta = y - prior[0]*x - prior[1]
        compatible = abs(delta) <= 3*_CAL_TOL
        shift = np.median(delta[compatible]) if np.any(compatible) else 0.
        curves.append((prior[0], prior[1] + shift))
    return np.asarray(curves)


def _features(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    feat = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        m = labs == lab
        r = ref[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        feat[m] = (feat[m] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    means = []
    for lab, stratum in np.unique(np.c_[labs, strata], axis=0):
        cell = feat[(labs == lab) & (strata == stratum)]
        n = np.isfinite(cell).sum(axis=0)
        means.append(np.divide(np.nansum(cell, axis=0), n, out=np.full(4, np.nan), where=n > 0))
    means = np.asarray(means)
    n = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), n, out=np.zeros(4), where=n > 0)
    return np.c_[np.ones(len(feat)), np.where(np.isfinite(feat), feat, fill)]


def _inputs(split, calibration, dt, threshold, feat):
    gain, bias = calibration[split['lab_id']].T
    past = gain[:, None]*split['past_stress_cmd'] + bias[:, None]
    hold = gain*split['stress_hold_cmd'] + bias
    ov = np.maximum(abs(hold) - threshold, 0.)**2 * np.sign(hold)
    work = (np.maximum(abs(past) - threshold, 0.)**2 @ dt) * hold
    return past, hold, feat, ov, work


def _design(inputs, dt, times, logs, deriv=False):
    past, hold, feat, ov, work = inputs
    tau = np.exp(logs)
    fast = np.argmin(logs)
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a, b = after[:, None]/tau, dt[:, None]/tau
    ea, eb = np.exp(-a), np.exp(-b)
    kernel = ea * -np.expm1(-b)
    d = hold[:, None] - past @ kernel
    u = times[:, None]/tau
    e, f = np.exp(-u), -np.expm1(-u)
    A = np.empty((len(past), len(times)+1, 9))
    A[:, :-1, :2] = d[:, None, :]*f
    A[:, :-1, 2:7] = feat[:, None, :]*times[None, :, None]
    A[:, :-1, 7] = ov[:, None]*times
    A[:, :-1, 8] = work[:, None]*f[:, fast]
    A[:, -1] = np.c_[d/tau, feat, ov, work/tau[fast]]
    if not deriv:
        return A
    dd = -past @ (a*kernel - ea*b*eb)
    G = np.concatenate((dd[:, None, :]*f - d[:, None, :]*u*e, ((dd-d)/tau)[:, None, :]), axis=1)
    J = work[:, None]*np.r_[-u[:, fast]*e[:, fast], -1./tau[fast]]
    return A, G, J, fast


def _fit(inputs, dt, times, y):
    valid = np.isfinite(y)
    yy = y[valid]
    scale = np.broadcast_to(np.r_[np.ones(len(times)), 1.5], y.shape)[valid]

    def fun(p):
        A = _design(inputs, dt, times, p[:2])[valid]
        return (np.einsum('ij,j->i', A, p[2:]) - yy)/scale

    def jac(p):
        A, G, J, fast = _design(inputs, dt, times, p[:2], True)
        G = G[valid]*p[2:4]
        G[:, fast] += J[valid]*p[-1]
        return np.c_[G, A[valid]]/scale[:, None]

    grid = np.geomspace(.18, 12., 8)
    starts = []
    for i, a in enumerate(grid[:-1]):
        for b in grid[i+1:]:
            logs = np.log([a, b])
            A = _design(inputs, dt, times, logs)[valid]/scale[:, None]
            z = yy/scale
            w = np.ones(len(z))
            for _ in range(5):
                c = np.linalg.lstsq(A*w[:, None], z*w, rcond=None)[0]
                r = np.einsum('ij,j->i', A, c)-z
                s = max(1e-4, 1.4826*np.median(abs(r-np.median(r))))
                w = 1./np.hypot(1., r/(2*s))
            starts.append((np.sum(np.log1p((r/.035)**2)), np.r_[logs, c]))
    bounds = (np.r_[np.log([.07, .07]), np.full(9, -np.inf)], np.r_[np.log([24., 24.]), np.full(9, np.inf)])

    def fit(start, sigma):
        return least_squares(fun, start, jac=jac, bounds=bounds, loss='cauchy', f_scale=sigma, x_scale='jac', max_nfev=140)

    pilots = [fit(p, .035) for _, p in sorted(starts, key=lambda z: z[0])[:4]]
    p = min(pilots, key=lambda z: z.cost).x
    residual = _design(inputs, dt, times, p[:2]) @ p[2:] - y
    def mad(x):
        return max(1e-5, 1.4826*np.nanmedian(abs(x-np.nanmedian(x))))
    scale = np.broadcast_to(np.r_[np.full(len(times), mad(residual[:, :-1])), mad(residual[:, -1])], y.shape)[valid]
    p = min((fit(q.x, 2.) for q in pilots), key=lambda z: z.cost).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def _condition_drift(train, test, train_inputs, train_y, prediction, dt, times, taus, coef):
    # Missing coupon features affect the linear drift only. Update that
    # component using the supplied noisy onset measurement, not the Prony state.
    missing = ~np.isfinite(test['coupon_features_raw'])
    use = missing.any(axis=1) & np.isfinite(test['instant_rate_obs'])
    if not np.any(use):
        return
    observed = np.isfinite(train['coupon_features_raw'])
    complete = observed.all(axis=1)
    rate = _design(train_inputs, dt, times[:0], np.log(taus))[:, 0] @ coef
    residual = (train_y[:, -1] - rate)[complete & np.isfinite(train_y[:, -1])]
    if not residual.size:
        return
    noise = max(1e-5, 1.4826*np.median(abs(residual - np.median(residual))))
    aligned = np.where(observed, train_inputs[2][:, 1:], np.nan)
    variance = np.nan_to_num(np.nanvar(aligned, axis=0)) * coef[3:7]**2
    uncertainty = missing[use] @ variance
    gain = uncertainty/(uncertainty + noise**2)
    drift = gain*(test['instant_rate_obs'][use] - prediction[use, -1])
    prediction[use] += drift[:, None]*np.r_[times, 1.]


def solve_alpha_w02_batch_05(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    cal = _calibrate(dataset['calibration_load_cells'])
    feat = _features(train, test)
    n = len(train['lab_id'])
    threshold = dataset['yield_stress']
    tr = _inputs(train, cal, dt, threshold, feat[:n])
    te = _inputs(test, cal, dt, threshold, feat[n:])
    y = np.c_[train['strain_curves_obs']-train['strain_0'][:, None], train['instant_rate_obs']]
    taus, c = _fit(tr, dt, times, y)
    if not np.any(tr[3]):
        c[7] = 0.
    if not np.any(tr[4]):
        c[8] = 0.
    pred = _design(te, dt, times, np.log(taus)) @ c
    _condition_drift(train, test, tr, y, pred, dt, times, taus, c)
    return dict(estimated_taus=taus, hereditary_weights=c[:7], viscoplastic_weights=c[7:],
                predicted_strain=pred[:, :-1]+test['strain_0'][:, None], initial_rate_pred=pred[:, -1])
