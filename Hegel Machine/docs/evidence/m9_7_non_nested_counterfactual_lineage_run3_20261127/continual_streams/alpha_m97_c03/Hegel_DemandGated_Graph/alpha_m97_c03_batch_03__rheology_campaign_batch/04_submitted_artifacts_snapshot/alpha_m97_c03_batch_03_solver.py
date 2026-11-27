"""Estimate two-branch polymer memory and jammed slip from calibrated measurements."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _calibration_rows(sweep, channel):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    keep = (np.asarray(sweep['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _clean_sweep(x, y):
    """Find a clean affine sweep if ordinary least squares flags an anomaly."""
    fit = np.polyfit(x, y, 1)
    if np.max(np.abs(y - np.polyval(fit, x))) <= .01 or len(x) < 3:
        return fit, np.ones(len(x), dtype=bool), False
    candidates = []
    for i, j in zip(*np.triu_indices(len(x), k=1)):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        residual = np.abs((y - y[i]) - gain * (x - x[i]))
        keep = residual <= .01
        if keep.sum() >= 2 and np.any(residual > 3.):
            candidates.append(keep)
    if not candidates:
        return fit, np.ones(len(x), dtype=bool), False
    count = max(s.sum() for s in candidates)
    best = np.unique([s for s in candidates if s.sum() == count], axis=0)
    if len(best) != 1:
        return fit, np.ones(len(x), dtype=bool), True
    keep = best[0]
    return np.polyfit(x[keep], y[keep], 1), keep, True


def _archive_fit(channel):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    for sweep in reversed(history):
        x, y = _calibration_rows(sweep, channel)
        if len(x) >= 2 and np.ptp(x) > 0:
            fit, keep, emi = _clean_sweep(x, y)
            if not emi or (not keep.all() and keep.sum() >= 4):
                return fit
    raise ValueError(f'No usable archived load-cell calibration for channel {channel}')


def _calibrate(sweep):
    fits = []
    for channel in range(4):
        x, y = _calibration_rows(sweep, channel)
        if len(x) < 2:
            fit = _archive_fit(channel)
        else:
            fit, keep, emi = _clean_sweep(x, y)
            if emi and (keep.all() or keep.sum() < 4):
                fit = _archive_fit(channel)
                keep = np.abs(y - np.polyval(fit, x)) <= 3.
                if keep.sum() >= 4:
                    fit = np.polyfit(x[keep], y[keep], 1)
        fits.append(fit)
    return np.asarray(fits)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def joined(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, stratum = joined('lab_id'), joined('stratum')
    ref, aligned = joined('ref_coupons'), joined('coupon_features_raw').astype(float)
    for channel in np.unique(lab):
        rows = lab == channel
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - center) / scale
    cells = np.column_stack((lab, stratum))
    means = []
    for cell in np.unique(cells, axis=0):
        values = aligned[np.all(cells == cell, axis=1)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(aligned.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isnan(aligned), np.nanmean(means, axis=0), aligned)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = joined('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = joined('stress_hold_cmd') * gain + bias
    drift = np.column_stack((np.ones(len(lab)), aligned))
    return past, hold, drift, joined('strain_0')


def _design(taus, past, hold, drift, dt, times):
    # Closed form of the zero-initialized history recurrence.
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = past @ (np.exp(-later[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    gaps = hold[:, None] - memory
    relax = -np.expm1(-times[:, None] / taus)
    flow = times / (1. + .85 * times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    contrast = gaps[:, 0] - gaps[:, 1]
    curves = np.concatenate((gaps[:, None, :] * relax[None, :, :],
                             drift[:, None, :] * flow[None, :, None],
                             slip[:, None, None] * flow[None, :, None],
                             contrast[:, None, None] * relax[None, :, :1]), axis=2)
    rates = np.column_stack((gaps / taus, drift, slip, contrast / taus[0]))
    return curves, rates


def _fit(past, hold, drift, dt, times, curves, rates):
    ok = np.isfinite(curves)
    rate_ok = np.isfinite(rates)
    target = np.r_[curves[ok], rates[rate_ok]]
    n_curve = ok.sum()

    def matrix(logtau):
        a, b = _design(np.exp(logtau), past, hold, drift, dt, times)
        return np.vstack((a[ok], b[rate_ok]))

    def scales(residual):
        def mad(values):
            return max(1e-6, 1.4826 * np.median(np.abs(values - np.median(values))))
        curve = mad(residual[:n_curve])
        rate = mad(residual[n_curve:]) if rate_ok.any() else curve
        return np.r_[np.full(n_curve, curve), np.full(rate_ok.sum(), rate)]

    lower = np.log(min(times[times > 0].min(), dt[dt > 0].min()) / 20.)
    upper = np.log(20. * max(times.max(), dt.sum()))
    bounds = (np.r_[lower, lower, np.full(9, -np.inf)],
              np.r_[upper, upper, np.full(9, np.inf)])
    seeds = times.max() * np.array([[.05, .28], [.08, .65], [.13, 1.6], [.24, .65], [.025, 2.]])
    starts = []
    # The slip relaxation is attached to branch 0: never sort a fitted pair.
    for tau in np.vstack((seeds, seeds[:, ::-1])):
        logtau = np.clip(np.log(tau), lower, upper)
        weights = np.linalg.lstsq(matrix(logtau), target, rcond=None)[0]
        starts.append(np.r_[logtau, weights])

    def residual(p):
        return matrix(p[:2]) @ p[2:] - target

    def optimize(p, scale, loss):
        return least_squares(lambda x: residual(x) / scale, p, bounds=bounds,
                             loss=loss, x_scale='jac', max_nfev=150)

    scale = scales(residual(min(starts, key=lambda p: np.linalg.norm(residual(p)))))
    pilot = min((optimize(p, scale, 'soft_l1') for p in starts), key=lambda fit: fit.cost)
    scale = 2. * scales(residual(pilot.x))
    fit = min((optimize(p, scale, 'cauchy') for p in [pilot.x, *starts]), key=lambda f: f.cost)
    fit = optimize(fit.x, 2. * scales(residual(fit.x)), 'cauchy')
    return np.exp(fit.x[:2]), fit.x[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    past, hold, drift, baseline = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    train = dataset['train']
    n = len(train['lab_id'])
    taus, weights = _fit(past[:n], hold[:n], drift[:n], dt, times,
                         np.asarray(train['strain_curves_obs']) - baseline[:n, None],
                         np.asarray(train['instant_rate_obs']))
    curves, rates = _design(taus, past[n:], hold[n:], drift[n:], dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': baseline[n:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
