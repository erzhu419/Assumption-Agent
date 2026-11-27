"""Calibrated two-branch creep in the shear-banding jammed-slip regime."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for table in reversed(history):
        take = np.asarray(table['lab_id']) == channel
        x = np.asarray(table['cmd_stress'])[take]
        if np.unique(x).size >= 2:
            return np.polyfit(x, np.asarray(table['true_stress'])[take], 1)
    return None


def _calibrate(table):
    lab, cmd, true = (np.asarray(table[key]) for key in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        i, j = np.triu_indices(x.size, k=1)
        distinct = x[i] != x[j]
        i, j = i[distinct], j[distinct]
        spike = False
        if i.size:
            # Choose the line with the most near-collinear points before
            # testing for EMI. A spike can skew the median slope of four rows.
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            offsets = y[i] - slopes*x[i]
            residuals = np.abs(y - slopes[:, None]*x - offsets[:, None])
            near = residuals <= 0.1
            support = near.sum(axis=1)
            candidates = np.flatnonzero(support == support.max())
            best = candidates[np.argmin(np.sum(np.where(near[candidates], residuals[candidates], 0.), axis=1))]
            gain, bias = np.polyfit(x[near[best]], y[near[best]], 1)
            clean = np.abs(y - (gain*x + bias)) <= 3.0
            spike = not np.all(clean)
        else:
            clean = np.ones(x.size, dtype=bool)
        # A historical line can mistake calibration drift for EMI. Keep a
        # supported in-batch fit and consult the archive only for fallback.
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
            continue
        archive = _archived_calibration(channel) if x.size < 2 or spike else None
        if archive is not None:
            result[channel] = archive
        elif np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        else:
            result[channel] = (1.0, np.median(y - x) if y.size else 0.0)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    keys = ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons',
            'past_stress_cmd', 'stress_hold_cmd', 'strain_0')
    lab, stratum, x, ref, past, hold, strain0 = (
        np.concatenate((train[key], test[key])) for key in keys
    )
    x = x.astype(float, copy=False)
    for channel in np.unique(lab):
        mask = lab == channel
        panel = ref[mask]
        panel = panel[:, np.any(np.isfinite(panel), axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    pooled = np.nanmean(means, axis=0)
    for group in np.unique(stratum):
        mask = stratum == group
        donors = means[cells[:, 1] == group]
        count = np.isfinite(donors).sum(axis=0)
        fill = np.divide(np.nansum(donors, axis=0), count, out=pooled.copy(), where=count > 0)
        x[mask] = np.where(np.isfinite(x[mask]), x[mask], fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _design(log_tau, past, hold, features, dt, times):
    tau = np.exp(log_tau)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    gap = hold[:, None] - past @ kernel
    step = -np.expm1(-times[:, None] / tau)
    terminal = times / (1.0 + 0.85 * times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    coupling = gap[:, 0] - gap[:, 1]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * terminal[None, :, None],
                            (slip[:, None] * terminal)[:, :, None],
                            (coupling[:, None] * step[:, 0])[:, :, None]), axis=2)
    rate = np.column_stack((gap / tau, features, slip, coupling / tau[0]))
    return curve, rate


def _fit(past, hold, features, dt, times, curves, rates):
    curve_ok, rate_ok = np.isfinite(curves), np.isfinite(rates)
    y = np.r_[curves[curve_ok], rates[rate_ok]]
    nc, nr = curve_ok.sum(), rate_ok.sum()

    def design(log_tau):
        curve, rate = _design(log_tau, past, hold, features, dt, times)
        return np.vstack((curve[curve_ok], rate[rate_ok]))

    def profile(log_tau, row_weight):
        a = design(log_tau) * row_weight[:, None]
        b = y * row_weight
        w = lstsq(a, b, lapack_driver='gelsy', check_finite=False)[0]
        return a @ w - b, w

    limits = np.log([0.025, 80.0])
    balance = np.r_[np.ones(nc), np.full(nr, np.sqrt(curves.shape[1]))]
    starts = ((0.25, 2.5), (0.7, 5.0), (2.5, 0.25), (5.0, 0.7))
    projected = []
    for start in starts:
        fit = least_squares(lambda t: profile(t, balance)[0], np.log(start),
                            bounds=limits, max_nfev=65)
        residual, w = profile(fit.x, balance)
        projected.append((residual @ residual, np.r_[fit.x, w]))
    projected.sort(key=lambda item: item[0])

    def raw(p):
        return design(p[:2]) @ p[2:] - y

    def mad(x):
        return max(1e-7, 1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 1.0

    r = raw(projected[0][1])
    scale = np.r_[np.full(nc, mad(r[:nc])), np.full(nr, mad(r[nc:]))]
    bounds = (np.r_[np.full(2, limits[0]), np.full(9, -np.inf)],
              np.r_[np.full(2, limits[1]), np.full(9, np.inf)])
    fit = min((least_squares(lambda p: raw(p) / scale, start, bounds=bounds,
                             loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=110)
               for _, start in projected[:3]), key=lambda result: result.cost)
    r = raw(fit.x)
    scale = np.r_[np.full(nc, mad(r[:nc])), np.full(nr, mad(r[nc:]))]
    fit = least_squares(lambda p: raw(p) / scale, fit.x, bounds=bounds,
                        loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=75)
    return fit.x[:2], fit.x[2:]


def solve_alpha_m97_c02_batch_06(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, weights = _fit(past[:n], hold[:n], features[:n], dt, times, curves, rates)
    curve, rate = _design(log_tau, past[n:], hold[n:], features[n:], dt, times)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
