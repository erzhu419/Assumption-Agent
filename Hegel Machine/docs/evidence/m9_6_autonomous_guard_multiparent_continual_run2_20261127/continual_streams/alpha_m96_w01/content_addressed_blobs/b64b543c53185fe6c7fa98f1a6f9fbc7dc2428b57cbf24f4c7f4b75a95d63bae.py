import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
_CHECK_TOL = 0.001


def _checks(panel, lab):
    x, y = map(np.asarray, (panel['cmd_stress'], panel['true_stress']))
    use = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _fit_line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(checks):
    try:
        with _HISTORY.open() as stream:
            history = json.load(stream)
    except FileNotFoundError:
        history = []
    result = []
    for lab in range(4):
        x, y = _checks(checks, lab)
        prior = None
        for sweep in reversed(history):
            u, v = _checks(sweep, lab)
            i, j = np.triu_indices(len(u), 1)
            good = abs(u[i] - u[j]) > .1
            if good.any():
                g = np.median((v[i[good]] - v[j[good]]) / (u[i[good]] - u[j[good]]))
                b = np.median(v - g*u)
                keep = abs(v - g*u - b) <= _CHECK_TOL
                if keep.sum() >= 2 and np.ptp(u[keep]) > .1:
                    prior = _fit_line(u[keep], v[keep])
                    break
        i, j = np.triu_indices(len(x), 1)
        good = abs(x[i] - x[j]) > .1
        i, j = i[good], j[good]
        g = (y[j] - y[i]) / (x[j] - x[i])
        candidates = list(zip(g[g > 0], (y[i] - g*x[i])[g > 0]))
        if prior is not None:
            candidates.insert(0, prior)
        if not candidates:
            result.append((1., np.median(y - x) if len(x) else 0.))
            continue
        def score(p):
            e = abs(y - p[0]*x - p[1])
            keep = e <= _CHECK_TOL
            return keep.sum(), -np.sum(e[keep]**2)
        best = max(candidates, key=score)
        keep = abs(y - best[0]*x - best[1]) <= _CHECK_TOL
        if keep.sum() >= 3 and np.ptp(x[keep]) > .1:
            # Do not extrapolate a narrow, sparse sweep that agrees with the
            # archived map to within measurement noise. Clear local drift wins.
            if (prior is not None and keep.sum() <= 4 and np.ptp(x[keep]) < .75
                    and np.max(abs(y[keep] - (prior[0]*x[keep] + prior[1]))) <= _CHECK_TOL / 2):
                result.append(prior)
            else:
                result.append(_fit_line(x[keep], y[keep]))
        elif prior is not None:
            # A narrow or sparse local sweep cannot reliably separate offset,
            # gain, and a bad check. Use the last verified same-lab sweep.
            result.append(prior)
        else:
            result.append(best)
    return np.asarray(result).T


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.empty_like(raw, dtype=float)
    for j in np.unique(lab):
        use = lab == j
        panel = ref[use]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        shift = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (raw[use] - shift) / scale
    missing = ~np.isfinite(x)
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    w = 1. / counts[cell]
    denom = (~missing).T @ w
    means = np.divide(np.where(missing, 0., x).T @ w, denom,
                      out=np.zeros(x.shape[1]), where=denom > 0)
    return np.column_stack((np.ones(len(x)), np.where(missing, means, x))), missing


def _design(tau, past, hold, dt, t, x, yield_stress):
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = np.exp(-trailing[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    delta = hold[:, None] - past @ kernel
    relax = -np.expm1(-t[:, None] / tau)
    ov = np.maximum(0., abs(hold) - yield_stress)**2 * np.sign(hold)
    work = (np.maximum(0., abs(past) - yield_stress)**2 @ dt) * hold
    a = np.concatenate((delta[:, None, :] * relax, x[:, None, :] * t[None, :, None],
                        ov[:, None, None] * t[None, :, None],
                        work[:, None, None] * relax[None, :, :1]), axis=2)
    r = np.column_stack((delta / tau, x, ov, work / tau[0]))
    return a, r


def _fit(past, hold, dt, t, x, y, rate, yield_stress):
    cm, rm = np.isfinite(y), np.isfinite(rate)
    yc, yr = y[cm], rate[rm]
    obs = np.r_[yc, yr]
    active = np.any(_design(np.array([.6, 2.]), past, hold, dt, t, x, yield_stress)[1] != 0, axis=0)
    def matrix(tau):
        a, r = _design(tau, past, hold, dt, t, x, yield_stress)
        return np.vstack((a[cm], r[rm]))[:, active]
    scales = np.ones(len(obs)) * .035
    starts = []
    grid = (.18, .32, .55, .9, 1.5, 2.5, 4.2, 7., 12.)
    for i, u in enumerate(grid[:-1]):
        for v in grid[i+1:]:
            mat = matrix(np.array([u, v]))
            theta = np.linalg.lstsq(mat, obs, rcond=None)[0]
            e = np.einsum('ij,j->i', mat, theta) - obs
            w = np.sqrt(np.minimum(1., max(.006, 2.5*np.median(abs(e))) / np.maximum(abs(e), 1e-12)))
            theta = np.linalg.lstsq(mat*w[:, None], obs*w, rcond=None)[0]
            e = np.einsum('ij,j->i', mat, theta) - obs
            starts.append((np.log1p((e/scales)**2).sum(), np.r_[np.array([np.log(u), np.log((v-u)/(24.9-v))]), theta]))
    starts.sort(key=lambda pair: pair[0])
    # Ordered parametrization keeps work coupled to the fast branch throughout.
    def taus(p):
        u = np.exp(p[0])
        return np.array([u, u + (24.9 - u) / (1 + np.exp(-p[1]))])
    def residual(p):
        return (np.einsum('ij,j->i', matrix(taus(p)), p[2:]) - obs) / scales
    n = active.sum()
    bounds = (np.r_[[np.log(.051), -12.], np.full(n, -np.inf)],
              np.r_[[np.log(12.), 12.], np.full(n, np.inf)])
    def fit(p):
        return least_squares(residual, p, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=140)
    best = min((fit(p) for _, p in starts[:3]), key=lambda sol: sol.cost)
    e = residual(best.x) * scales
    for block in (slice(None, len(yc)), slice(len(yc), None)):
        z = e[block]
        if z.size:
            scales[block] = max(.001, 2.5*1.4826*np.median(abs(z - np.median(z))))
    best = fit(best.x)
    theta = np.zeros(9)
    theta[active] = best.x[2:]
    return taus(best.x), theta


def _stress(split, gain, bias):
    lab = np.asarray(split['lab_id'])
    return (split['past_stress_cmd']*gain[lab, None] + bias[lab, None],
            split['stress_hold_cmd']*gain[lab] + bias[lab])


def solve_alpha_m96_w01_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x, missing = _features(train, test)
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    dt, t = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    sy = float(dataset['yield_stress'])
    past, hold = _stress(train, gain, bias)
    n = len(hold)
    keep = ~missing[:n].any(axis=1)
    if keep.sum() < 12:
        keep[:] = True
    y = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    tau, theta = _fit(past[keep], hold[keep], dt, t, x[:n][keep], y[keep],
                      np.asarray(train['instant_rate_obs'])[keep], sy)
    past, hold = _stress(test, gain, bias)
    a, r = _design(tau, past, hold, dt, t, x[n:], sy)
    curve = np.asarray(test['strain_0'])[:, None] + a @ theta
    rate = r @ theta
    # A missing feature leaves only the linear creep slope unresolved.
    observed = np.asarray(test['instant_rate_obs'])
    update = missing[n:].any(axis=1) & np.isfinite(observed)
    curve[update] += (observed[update] - rate[update])[:, None] * t
    rate[update] = observed[update]
    return dict(estimated_taus=tau, hereditary_weights=theta[:7], viscoplastic_weights=theta[7:],
                predicted_strain=curve, initial_rate_pred=rate)
