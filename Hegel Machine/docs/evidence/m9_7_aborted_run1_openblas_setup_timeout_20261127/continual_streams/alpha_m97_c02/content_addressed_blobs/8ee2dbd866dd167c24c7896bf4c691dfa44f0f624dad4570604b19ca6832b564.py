"""Calibrated two-branch Prony model for jammed shear-banding slip."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for sweep in reversed(history):
        mask = np.asarray(sweep['lab_id']) == channel
        x, y = (np.asarray(sweep[key])[mask] for key in ('cmd_stress', 'true_stress'))
        if np.unique(x).size >= 2:
            return np.polyfit(x, y, 1)
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
    target = np.r_[curves.ravel(), rates]
    bounds = (np.log([0.015, 1.015]), np.log([20., 250.]))

    def design_at(params):
        taus = np.exp([params[0], params[0] + params[1]])
        curve, rate = _basis(taus, past, hold, features, dt, times)
        return np.vstack((curve.reshape(-1, 9), rate)), taus

    def project(params, weight):
        design, taus = design_at(params)
        coef = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)[0]
        return design @ coef - target, coef, taus

    # Reweight individual observations, so a transient spike cannot discard
    # the other time points in the same specimen's trace.
    best = None
    for fast, slow in ((0.2, 1.5), (0.55, 4.), (1.2, 8.)):
        params = np.log([fast, slow / fast])
        weight = np.ones_like(target)
        for scale in (0.20, 0.08, 0.04, 0.025, 0.025, 0.025, 0.025, 0.025):
            residual = project(params, weight)[0]
            weight = 1 / np.hypot(1, residual / scale)
            fit = least_squares(lambda p: weight * project(p, weight)[0], params,
                                bounds=bounds, max_nfev=45, ftol=1e-8, xtol=1e-8)
            params = fit.x
        residual, coef, taus = project(params, weight)
        cost = np.sum(np.log1p((residual / scale)**2))
        if best is None or cost < best[0]:
            best = cost, taus, coef
    return best[1:]


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
