"""Calibrated two-branch Prony model with a crystallization plateau."""

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
    for sweep in reversed(history):
        mask = np.asarray(sweep['lab_id']) == channel
        cmd = np.asarray(sweep['cmd_stress'])[mask]
        true = np.asarray(sweep['true_stress'])[mask]
        if np.unique(cmd).size >= 2:
            return np.polyfit(cmd, true, 1)
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
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        elif (x.size < 2 or spike) and (archive := _archived_calibration(channel)) is not None:
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


def _basis(taus, past, hold, features, dt, times, activity, plateau=True):
    # A stress increment is attenuated by all steps after its own step.
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    z = past @ (np.exp(-age[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    gap = hold[:, None] - z
    rise = -np.expm1(-times[:, None] / taus)
    # Shear banding selects the flow law; damage scales the two network terms.
    flow = np.log1p(times) if plateau else times
    cols = 9 if plateau else 7
    curves = np.empty((len(hold), len(times), cols))
    rates = np.empty((len(hold), cols))
    curves[:, :, :2] = gap[:, None, :] * rise
    rates[:, :2] = gap / taus
    curves[:, :, 2:7] = flow[None, :, None] * features[:, None, :]
    rates[:, 2:7] = features
    if plateau:
        extra = activity * hold
        network = activity * z[:, 0]
        curves[:, :, 7] = extra[:, None] * rise[:, 0]
        curves[:, :, 8] = network[:, None] * flow[None, :]
        rates[:, 7] = extra / taus[0]
        rates[:, 8] = network
    return curves, rates


def _activity(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.50:
        return np.zeros(len(past))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)
    return np.tanh(excess**2 @ dt)


def _fit(past, hold, features, dt, times, activity, observed, rates, plateau=True):
    ncols = 9 if plateau else 7
    nc = observed.size
    y = np.r_[observed.ravel(), rates]
    mask = np.isfinite(y)
    is_rate = (np.arange(y.size) >= nc)[mask]
    y = y[mask]

    def project(log_tau, weight):
        curves, derivatives = _basis(np.exp(log_tau), past, hold, features, dt, times, activity, plateau)
        A = np.concatenate((curves.reshape(-1, ncols), derivatives))[mask]
        coef = np.linalg.lstsq(A * weight[:, None], y * weight, rcond=None)[0]
        return A @ coef - y, coef

    lo, hi = np.log([0.025, 150.0])
    weight = np.where(is_rate, 1.0, 1.0 / np.sqrt(len(times)))

    def optimize(start):
        return least_squares(lambda p: project(p, weight)[0] * weight, start,
                             bounds=(lo, hi), max_nfev=70)

    starts = [(0.35, 2.0), (0.7, 4.0), (1.4, 8.0)]
    init = [optimize(np.log(s)) for s in starts]
    fit = min(init, key=lambda r: r.cost)
    for _ in range(12):
        residual, _ = project(fit.x, weight)
        wnew = np.ones_like(weight)
        for m in (is_rate, ~is_rate):
            if not np.any(m):
                continue
            r = residual[m]
            scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-5)
            wnew[m] = 1.0 / np.hypot(2.385 * scale, r)
        wnew /= np.max(wnew)
        weight = wnew
        fit = optimize(fit.x)
    _, coef = project(fit.x, weight)
    tau = np.exp(fit.x)
    if not plateau or not np.any(activity > 0):
        order = np.argsort(tau)
        tau = tau[order]
        coef[:2] = coef[order]
    if not plateau:
        coef = np.r_[coef, 0.0, 0.0]
    return tau, coef


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    dt, times = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    plateau = dataset['shear_banding_index'] <= 0.50
    activity = _activity(dataset, past, dt)
    # Imputation supplies prediction features, not measured training covariates.
    # Prefer complete coupons for identifying the shared relaxation spectrum.
    fit_rows = np.isfinite(train['coupon_features_raw']).all(axis=1)
    if fit_rows.sum() <= n / 2 or (np.any(activity[:n] > 0) and not np.any(activity[:n][fit_rows] > 0)):
        fit_rows = slice(None)
    tau, coef = _fit(past[:n][fit_rows], hold[:n][fit_rows], features[:n][fit_rows],
                     dt, times, activity[:n][fit_rows],
                     (train['strain_curves_obs'] - strain0[:n, None])[fit_rows],
                     train['instant_rate_obs'][fit_rows], plateau)
    curves, rates = _basis(tau, past[n:], hold[n:], features[n:], dt, times, activity[n:], plateau)
    prediction_coef = coef[:curves.shape[-1]]
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:], 'predicted_strain': strain0[n:, None] + curves @ prediction_coef,
            'initial_rate_pred': rates @ prediction_coef}
