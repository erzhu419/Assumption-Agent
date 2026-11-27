import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _clean(cmd, true):
    if len(cmd) < 3:
        return np.ones(len(cmd), dtype=bool)
    i, j = np.triu_indices(len(cmd), 1)
    distinct = cmd[i] != cmd[j]
    slopes = (true[j[distinct]] - true[i[distinct]]) / (cmd[j[distinct]] - cmd[i[distinct]])
    offsets = true[:, None] - cmd[:, None] * slopes
    errors = np.abs(offsets - np.median(offsets, axis=0))
    return errors[:, np.argmin(np.median(errors, axis=0))] <= 3.0


def _calibration(cal, channel):
    m = cal['lab_id'] == channel
    cmd, true = cal['cmd_stress'][m], cal['true_stress'][m]
    clean = _clean(cmd, true)
    if len(cmd) < 2 or (not clean.all() and clean.sum() < 4):
        xs, ys = [], []
        for sweep in json.loads(_ARCHIVE.read_text()):
            take = np.asarray(sweep['lab_id']) == channel
            xs.extend(np.asarray(sweep['cmd_stress'])[take])
            ys.extend(np.asarray(sweep['true_stress'])[take])
        x, y = np.asarray(xs), np.asarray(ys)
        good = _clean(x, y)
        prior = np.polyfit(x[good], y[good], 1)
        if len(cmd) >= 2:
            clean = np.abs(true - np.polyval(prior, cmd)) <= 3.0
            if clean.sum() >= 2:
                return np.polyfit(cmd[clean], true[clean], 1)
        return prior
    return np.polyfit(cmd[clean], true[clean], 1)


def _prepare(d):
    train, test = d['train'], d['test']

    def join(k):
        return np.concatenate((train[k], test[k]), axis=0)

    lab, stratum = join('lab_id'), join('stratum')
    x, ref = join('coupon_features_raw'), join('ref_coupons')
    past, hold = join('past_stress_cmd'), join('stress_hold_cmd')
    cells = []
    for channel in np.unique(lab):
        m = lab == channel
        gain, bias = _calibration(d['calibration_load_cells'], channel)
        past[m] = gain * past[m] + bias
        hold[m] = gain * hold[m] + bias
        panel = ref[m]
        panel = panel[:, np.isfinite(panel).any(axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[m] = (x[m] - center) / scale
        cells.extend(m & (stratum == level) for level in np.unique(stratum[m]))
    means = []
    for cell in cells:
        values = x[cell]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    balanced = np.nanmean(means, axis=0)
    for cell, mean in zip(cells, means):
        x[cell] = np.where(np.isnan(x[cell]), np.where(np.isnan(mean), balanced, mean), x[cell])
    dt = np.asarray(d['past_dt'])
    damage = np.maximum(np.abs(past) - d['crit_stress'], 0.)**2 @ dt
    plateau = (d['shear_banding_index'] <= .50) & (damage > 0.)
    act = np.where(plateau, np.tanh(damage), 0.)
    return past, hold, np.column_stack((np.ones(len(x)), x)), act


def _basis(logtau, dt, t, data):
    past, hold, x, act = data
    tau = np.exp(logtau)
    after = (np.cumsum(dt[::-1])[::-1] - dt)[:, None]
    kernel = np.exp(-after / tau) * -np.expm1(-dt[:, None] / tau)
    z = past @ kernel
    gap = hold[:, None] - z
    step = -np.expm1(-t[:, None] / tau)
    # Held-out labs support logarithmic drift even at zero damage.
    phi = np.log1p(t)
    drift_terms = phi[None, :, None] * x[:, None, :]
    stress_term, memory_term = act * hold, act * z[:, 0]
    curves = np.concatenate((gap[:, None, :] * step, drift_terms,
                             (stress_term[:, None] * step[:, 0])[:, :, None],
                             (memory_term[:, None] * phi)[:, :, None]), axis=2)
    rates = np.column_stack((gap / tau, x, stress_term / tau[0], memory_term))
    return curves, rates


def _fit(dt, t, data, yc, yr):
    yc, yr = yc.ravel(), yr.ravel()
    target = np.r_[yc, yr]

    def design(logtau):
        c, r = _basis(logtau, dt, t, data)
        return np.vstack((c.reshape(-1, 9), r))

    # Profile out the linear coefficients to find initial relaxation times;
    # refine jointly using a robust loss for contaminated strain/rate readings.
    scale = np.r_[np.full(yc.size, .04), np.full(yr.size, .05)]
    lo, hi = np.log([.025, 120.])

    def linear(logtau):
        a = design(logtau) / scale[:, None]
        w = np.linalg.lstsq(a, target / scale, rcond=None)[0]
        return w, a @ w - target / scale

    def obj(logtau):
        return np.sum(linear(logtau)[1]**2)

    def raw(p):
        return np.einsum('ij,j->i', design(p[:2]), p[2:]) - target

    def residual(p):
        return raw(p) / scale

    bounds = (np.r_[[lo, lo], np.full(9, -np.inf)], np.r_[[hi, hi], np.full(9, np.inf)])
    opts = dict(bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=100)
    best = None
    for pair in ((.2, 2.), (.7, 5.), (1.5, 12.)):
        q = minimize(obj, np.log(pair), method='L-BFGS-B', bounds=[(lo, hi)]*2,
                     options={'maxiter': 45}).x
        # Branch zero is physically distinguished by both crystalline terms.
        p = np.r_[q, linear(q)[0]]
        fit = least_squares(residual, p, **opts)
        if best is None or fit.cost < best.cost:
            best = fit
    err = raw(best.x)
    for idx in (slice(0, yc.size), slice(yc.size, None)):
        e = err[idx]
        scale[idx] = max(1e-7, 2.5 * 1.4826 * np.median(np.abs(e - np.median(e))))
    p = least_squares(residual, best.x, **opts).x
    return p[:2], p[2:]


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    logtau, w = _fit(dt, t, tuple(a[:n] for a in data), yc, train['instant_rate_obs'])
    curves, rates = _basis(logtau, dt, t, tuple(a[n:] for a in data))
    return dict(estimated_taus=np.exp(logtau), hereditary_weights=w[:7],
                regime_mode_weights=w[7:], predicted_strain=test['strain_0'][:, None] + curves @ w,
                initial_rate_pred=rates @ w)
