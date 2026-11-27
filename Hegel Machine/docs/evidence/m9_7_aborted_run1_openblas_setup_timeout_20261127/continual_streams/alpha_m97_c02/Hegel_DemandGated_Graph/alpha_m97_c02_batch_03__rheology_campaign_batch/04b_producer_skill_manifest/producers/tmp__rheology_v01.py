"""Calibrated two-branch Prony model for jammed shear-banding slip."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    if isinstance(history, dict):
        history = history.get('channels', history.get('calibration_load_cells', history))
    if isinstance(history, dict) and 'lab_id' in history:
        if 'cmd_stress' in history and 'true_stress' in history:
            mask = np.asarray(history['lab_id']) == channel
            x = np.asarray(history['cmd_stress'])[mask]
            y = np.asarray(history['true_stress'])[mask]
            return np.polyfit(x, y, 1) if np.unique(x).size >= 2 else None
    if isinstance(history, dict):
        entry = history.get(str(channel))
    else:
        entry = next((row for row in reversed(history)
                      if isinstance(row, dict) and row.get('lab_id') == channel
                      and 'gain' in row and 'bias' in row), None)
    if entry is not None:
        return (entry['gain'], entry['bias']) if isinstance(entry, dict) else entry
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
            gain = np.median((y[i] - y[j]) / (x[i] - x[j]))
            offsets = y - gain*x
            clean = np.abs(offsets - np.median(offsets)) <= 3.0
            spike = not np.all(clean)
        else:
            clean = np.ones(x.size, dtype=bool)
        archive = _archived_calibration(channel) if x.size < 2 or spike else None
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        elif archive is not None:
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
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(taus, past, hold, features, dt, times):
    # Unroll the memory recurrence: each interval's contribution decays
    # through all later intervals before the start of the hold.
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = np.exp(-age[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    gap = hold[:, None] - past @ memory
    step = -np.expm1(-times[:, None] / taus)
    phi = times / (1 + 0.85 * times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    cross_gap = gap[:, 0] - gap[:, 1]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            (slip[:, None] * phi)[..., None],
                            (cross_gap[:, None] * step[:, 0])[..., None]), axis=2)
    rate = np.column_stack((gap / taus, features, slip, cross_gap / taus[0]))
    return curve, rate


def _fit(past, hold, features, dt, times, curves, rates):
    # The nine coefficients are linear conditional on the two time scales.
    # Use QR/SVD least squares rather than normal equations, since the
    # slip and Prony terms can have similar shapes on short holds.
    rate_weight = 1.0
    target = np.r_[curves.ravel(), rate_weight * rates]

    def project(params):
        taus = np.exp([params[0], params[0] + params[1]])
        curve, rate = _basis(taus, past, hold, features, dt, times)
        design = np.concatenate((curve.reshape(-1, 9), rate_weight * rate))
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        return design @ weights - target, weights, taus

    best = None
    bounds = (np.log([0.015, 1.015]), np.log([20., 250.]))
    for fast, slow in ((0.15, 1.), (0.4, 2.5), (0.8, 4.), (1.5, 8.)):
        fit = least_squares(lambda p: project(p)[0], np.log([fast, slow / fast]),
                            bounds=bounds, max_nfev=100,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or fit.cost < best.cost:
            best = fit
    _, weights, taus = project(best.x)
    return taus, weights


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, times,
                         train['strain_curves_obs'] - strain0[:n, None],
                         train['instant_rate_obs'])
    curve, rate = _basis(taus, past[n:], hold[n:], features[n:], dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
