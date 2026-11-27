import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = xc @ (y - y.mean()) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _spikes(x, y):
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool)
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    if not len(i):
        return np.zeros(len(x), dtype=bool)
    gains = (y[j] - y[i]) / (x[j] - x[i])
    offsets = y[:, None] - x[:, None] * gains
    errors = np.abs(offsets - np.median(offsets, axis=0))
    best = np.argmin(np.median(errors, axis=0))
    return errors[:, best] > 3.0


def _prior_line(channel):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        row = np.asarray(sweep['lab_id']) == channel
        x = np.asarray(sweep['cmd_stress'])[row]
        y = np.asarray(sweep['true_stress'])[row]
        keep = ~_spikes(x, y)
        if np.unique(x[keep]).size >= 2:
            return _line(x[keep], y[keep])
    raise ValueError(f'No usable calibration sweep for channel {channel}')


def _calibrate(x, y, channel):
    if len(x) >= 4:
        keep = ~_spikes(x, y)
        if keep.sum() >= 4 or keep.all():
            return _line(x[keep], y[keep])
    else:
        keep = np.ones(len(x), dtype=bool)
        if len(x) == 3:
            # Two surviving points cannot identify which of three rows spiked.
            # Look for a >3 discrepancy before consulting the prior sweep.
            errors = [abs(y[k] - np.polyval(_line(np.delete(x, k), np.delete(y, k)), x[k]))
                      for k in range(3) if np.unique(np.delete(x, k)).size == 2]
            slope, bias = _line(x, y)
            if not errors or max(errors) <= 3 or np.max(np.abs(y - slope*x - bias)) < .01:
                return slope, bias
        elif len(x) == 2:
            return _line(x, y)
    gain, bias = _prior_line(channel)
    if len(x) >= 3 and keep.all():
        keep = np.abs(y - (gain*x + bias)) <= 3.0
    if np.unique(x[keep]).size >= 2:
        clean_x, clean_y = x[keep], y[keep]
        fitted_gain, fitted_bias = _line(clean_x, clean_y)
        # With fewer than four clean readings after EMI rejection, borrow
        # the archived slope only if supported by the in-batch contrasts.
        # Keep the stress level anchored to the current measurements.
        x_centered = clean_x - clean_x.mean()
        if keep.sum() < 4:
            contrasts = clean_y - clean_y.mean() - gain * x_centered
            if np.ptp(contrasts) <= 0.001:
                return gain, clean_y.mean() - gain * clean_x.mean()
        return fitted_gain, fitted_bias
    return gain, bias


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd'])).astype(float)
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']].astype(float)
    for c in np.unique(lab):
        rows = np.asarray(cal['lab_id']) == c
        gain, bias = _calibrate(np.asarray(cal['cmd_stress'])[rows], np.asarray(cal['true_stress'])[rows], c)
        take = lab == c
        past[take] = gain * past[take] + bias
        hold[take] = gain * hold[take] + bias
        panel = ref[take]
        x[take] = (x[take] - np.nanmedian(np.nanmean(panel, axis=0))) / np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
    cells = []
    for c in np.unique(lab):
        for stratum_id in np.unique(stratum[lab == c]):
            take = (lab == c) & (stratum == stratum_id)
            valid = np.isfinite(x[take]).sum(axis=0)
            mean = np.divide(np.nansum(x[take], axis=0), valid, out=np.full(x.shape[1], np.nan), where=valid != 0)
            cells.append((c, take, mean))
    for c, take, mean in cells:
        if np.isnan(mean).any():
            with np.errstate(invalid='ignore', divide='ignore'):
                local = np.array([v for lab_id, _, v in cells if lab_id == c])
                count = np.isfinite(local).sum(axis=0)
                local = np.divide(np.nansum(local, axis=0), count, out=np.zeros(x.shape[1]), where=count != 0)
            mean = np.where(np.isnan(mean), local, mean)
        x[take] = np.where(np.isnan(x[take]), mean, x[take])
    return np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times']), past, hold, x


def _basis(logtau, dt, t, past, hold, x):
    tau = np.exp(logtau)
    h = dt[:, None] / tau
    a = np.exp(-h)
    b = -np.expm1(-h)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for s in range(len(dt)):
        u = past[:, s, None]
        dz = a[s] * (dz + h[s] * (z - u))
        z = a[s] * z + b[s] * u
    gap = hold[:, None] - z
    k = t[:, None] / tau
    phi = -np.expm1(-k)
    dphi = -k * np.exp(-k)
    drift = np.column_stack((np.ones(len(hold)), x))
    curves = np.concatenate((gap[:, None, :] * phi, drift[:, None, :] * t[None, :, None]), axis=2)
    rates = np.column_stack((gap / tau, drift))
    dc = -dz[:, None, :] * phi + gap[:, None, :] * dphi
    dr = (-dz - gap) / tau
    return curves, rates, dc, dr


def _fit(dt, t, past, hold, x, yc, yr, complete=None):
    y = np.r_[yc.ravel(), yr]
    n = yc.size
    low = np.log(min(dt.min(), t[t > 0].min()) / 20)
    high = np.log((dt.sum() + t.max()) * 20)
    bounds = (np.r_[[low]*2, [-np.inf]*7], np.r_[[high]*2, [np.inf]*7])
    scale = np.ones_like(y)
    last = None

    def design(u):
        nonlocal last
        if last is None or not np.array_equal(u, last[0]):
            c, r, dc, dr = _basis(u, dt, t, past, hold, x)
            last = (u.copy(), np.vstack((c.reshape(-1, 7), r)), np.vstack((dc.reshape(-1, 2), dr)))
        return last[1:]

    def fun(p):
        a, _ = design(p[:2])
        return (np.einsum('ij,j->i', a, p[2:]) - y) / scale

    def jac(p):
        a, d = design(p[:2])
        return np.column_stack((d * p[2:4], a)) / scale[:, None]

    starts = []
    for a, b in ((.25, 1.2), (.5, 3), (1., 5.), (2., 10.)):
        u = np.clip(np.log([a, b]), low + 1e-6, high - 1e-6)
        matrix, _ = design(u)
        w = np.linalg.lstsq(matrix, y, rcond=None)[0]
        starts.append(least_squares(fun, np.r_[u, w], jac=jac, x_scale='jac', bounds=bounds, max_nfev=80))
    best = min(starts, key=lambda r: r.cost)
    residual = fun(best.x)
    for sl in (slice(n), slice(n, None)):
        v = residual[sl]
        scale[sl] = max(1.4826 * np.median(np.abs(v - np.median(v))), 1e-6)
    p = least_squares(fun, best.x, jac=jac, loss='soft_l1', f_scale=2.5,
                      x_scale='jac', bounds=bounds, max_nfev=80).x
    # Imputation is required for predictions, but imputed training rows
    # need not determine the physical parameters when there are enough
    # complete rows to identify all seven linear coefficients.
    if complete is None:
        complete = np.ones(len(hold), dtype=bool)
    rows = np.r_[np.repeat(complete, len(t)), complete]
    a, _ = design(p[:2])
    if np.linalg.matrix_rank(a[rows]) < 7:
        rows[:] = True
    residual = np.einsum('ij,j->i', a, p[2:]) - y
    by_target = np.column_stack((residual[:n].reshape(yc.shape), residual[n:]))
    v = by_target[rows[n:]]
    target_scale = np.maximum(1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0), 1e-6)
    scale[:] = np.r_[np.tile(target_scale[:-1], len(hold)), np.full(len(hold), target_scale[-1])]
    p = least_squares(lambda v: fun(v)[rows], p, jac=lambda v: jac(v)[rows],
                      loss='cauchy', f_scale=2.5, x_scale='jac', bounds=bounds, max_nfev=80).x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p


def solve_alpha_m97_c01_batch_02(dataset: dict) -> dict:
    dt, t, past, hold, x = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    p = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'], complete)
    curves, rates, _, _ = _basis(p[:2], dt, t, past[n:], hold[n:], x[n:])
    return dict(estimated_taus=np.exp(p[:2]), hereditary_weights=p[2:], regime_mode_weights=np.zeros(2),
                predicted_strain=test['strain_0'][:, None] + curves @ p[2:], initial_rate_pred=rates @ p[2:])
