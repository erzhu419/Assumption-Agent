"""Calibrate the load cells and fit the two-branch jammed-slip response."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = (xc @ (y - y.mean())) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _spikes(x, y):
    i, j = np.triu_indices(len(x), 1)
    different = x[i] != x[j]
    i, j = i[different], j[different]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    offsets = y[:, None] - x[:, None] * slopes
    errors = np.abs(offsets - np.median(offsets, axis=0))
    return errors[:, np.argmin(np.median(errors, axis=0))] > 3.0


def _prior_line(channel):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        take = np.asarray(sweep['lab_id']) == channel
        x = np.asarray(sweep['cmd_stress'])[take]
        y = np.asarray(sweep['true_stress'])[take]
        if len(x) >= 4:
            keep = ~_spikes(x, y)
            x, y = x[keep], y[keep]
        if np.unique(x).size >= 2:
            return _line(x, y)
    raise ValueError(f'No usable calibration sweep for channel {channel}')


def _calibrate(x, y, channel):
    if len(x) >= 4:
        keep = ~_spikes(x, y)
        if keep.sum() >= 4 or keep.all():
            return _line(x[keep], y[keep])
    elif len(x) == 3:
        gain, bias = _line(x, y)
        errors = [abs(y[k] - np.polyval(_line(x[np.arange(3) != k], y[np.arange(3) != k]), x[k]))
                  for k in range(3) if np.unique(x[np.arange(3) != k]).size == 2]
        if max(errors, default=0) <= 3 or np.max(np.abs(y - (gain*x + bias))) < 0.01:
            return gain, bias
    elif len(x) == 2:
        return _line(x, y)

    gain, bias = _prior_line(channel)
    clean = np.abs(y - (gain * x + bias)) <= 3.0
    if np.unique(x[clean]).size >= 2:
        return _line(x[clean], y[clean])
    if clean.any():
        bias = np.mean(y[clean] - gain * x[clean])
    return gain, bias


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    x = raw.astype(float)
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd'])).astype(float)
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']].astype(float)
    cal = dataset['calibration_load_cells']
    for channel in np.unique(lab):
        rows = np.asarray(cal['lab_id']) == channel
        cx, cy = (np.asarray(cal[k])[rows] for k in ('cmd_stress', 'true_stress'))
        finite = np.isfinite(cx) & np.isfinite(cy)
        gain, bias = _calibrate(cx[finite], cy[finite], channel)
        take = lab == channel
        past[take] = gain * past[take] + bias
        hold[take] = gain * hold[take] + bias
        panel = ref[take]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[take] = (x[take] - center) / scale
        cells = [(take & (stratum == s)) for s in np.unique(stratum[take])]
        means = []
        for cell in cells:
            count = np.isfinite(x[cell]).sum(axis=0)
            means.append(np.divide(np.nansum(x[cell], axis=0), count,
                                   out=np.full(x.shape[1], np.nan), where=count > 0))
        means = np.asarray(means)
        count = np.isfinite(means).sum(axis=0)
        balanced = np.divide(np.nansum(means, axis=0), count,
                             out=np.zeros(x.shape[1]), where=count > 0)
        for cell, mean in zip(cells, means):
            fill = np.where(np.isfinite(mean), mean, balanced)
            x[cell] = np.where(np.isfinite(x[cell]), x[cell], fill)
    return past, hold, x, np.isfinite(raw).all(axis=1)


def _design(tau, dt, time, past, hold, x):
    decay = np.exp(-dt[:, None] / tau)
    step = -np.expm1(-dt[:, None] / tau)
    memory = np.zeros((len(hold), 2))
    for j in range(len(dt)):
        memory = decay[j] * memory + step[j] * past[:, j, None]
    gap = hold[:, None] - memory
    transient = -np.expm1(-time[:, None] / tau)
    phi = time / (1 + 0.85 * time)
    drift = np.column_stack((np.ones(len(x)), x, np.sign(hold) * np.sqrt(np.abs(hold))))
    cross = gap[:, 0] - gap[:, 1]
    curves = np.concatenate((gap[:, None, :] * transient[None, :, :],
                             drift[:, None, :] * phi[None, :, None],
                             cross[:, None, None] * transient[None, :, :1]), axis=2)
    rates = np.column_stack((gap / tau, drift, cross / tau[0]))
    return curves, rates


def _fit(dt, time, past, hold, x, yc, yr):
    ic, ir = np.isfinite(yc), np.isfinite(yr)
    target = np.r_[yc[ic], yr[ir]]
    n_curve = ic.sum()
    bounds = (np.log(0.025), np.log(100.0))

    def design(u):
        c, r = _design(np.exp(u), dt, time, past, hold, x)
        return np.vstack((c[ic], r[ir]))

    def profile(u, scale, robust):
        a = design(u) / scale[:, None]
        y = target / scale
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        if robust:
            for _ in range(8):
                e = np.einsum('ij,j->i', a, w) - y
                q = (1.0 + (e / 2.5)**2)**-0.25
                w = np.linalg.lstsq(a*q[:, None], y*q, rcond=None)[0]
        e = np.einsum('ij,j->i', a, w) - y
        if robust:
            # Signed square root of the pseudo-Huber objective.
            e = e * np.sqrt(2.0 / (1.0 + np.hypot(1.0, e / 2.5)))
        return e, w

    scale = np.ones(len(target))
    pilot = min((least_squares(lambda u: profile(u, scale, False)[0], np.log(start), bounds=bounds)
                 for start in ((0.35, 2.5), (0.8, 5.0))), key=lambda fit: fit.cost)
    residual = profile(pilot.x, scale, False)[0]
    for sl in (slice(None, n_curve), slice(n_curve, None)):
        e = residual[sl]
        if len(e):
            scale[sl] = max(1.4826 * np.median(np.abs(e - np.median(e))), 1e-7)
    fit = least_squares(lambda u: profile(u, scale, True)[0], pilot.x, bounds=bounds)
    # Identify isolated curve corruption using only the current fit's training rows.
    a = design(fit.x)
    w = profile(fit.x, scale, True)[1]
    residual = (a @ w - target) / scale
    keep = np.ones(len(target), dtype=bool)
    keep[:n_curve] = np.abs(residual[:n_curve]) <= 5.0
    y = target[keep] / scale[keep]

    def clean_profile(u):
        a = design(u)[keep] / scale[keep, None]
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        return a @ w - y, w

    refit = least_squares(lambda u: clean_profile(u)[0], fit.x, bounds=bounds)
    return np.exp(refit.x), clean_profile(refit.x)[1]


def solve_alpha_m97_c01_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    past, hold, x, complete = _prepare(dataset)
    n = len(train['lab_id'])
    # Fit on measured coupons; imputed feature values are used for prediction.
    keep = complete[:n]
    yc = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    yr = np.asarray(train['instant_rate_obs'])
    tau, weights = _fit(dt, time, past[:n][keep], hold[:n][keep], x[:n][keep], yc[keep], yr[keep])
    curves, rates = _design(tau, dt, time, past[n:], hold[n:], x[n:])
    return {'estimated_taus': tau,
            'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
