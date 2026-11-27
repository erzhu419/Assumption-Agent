"""Calibrated two-branch creep with a strain-crystallization plateau."""
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


def _design(log_tau, past, hold, features, dt, times, act):
    tau = np.exp(log_tau)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    z = past @ (np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau))
    gap = hold[:, None] - z
    step = -np.expm1(-times[:, None] / tau)
    phi = np.log1p(times)
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            (act * hold)[:, None, None] * step[None, :, :1],
                            (act * z[:, 0])[:, None, None] * phi[None, :, None]), axis=2)
    rate = np.column_stack((gap / tau, features, act * hold / tau[0], act * z[:, 0]))
    return curve, rate


def _activation(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.50:
        return np.zeros(len(past))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
    return np.tanh(excess**2 @ dt)


def _profile_fit(past, hold, features, dt, times, act, curves, rates, rate_weight=1.):
    curve_ok, rate_ok = np.isfinite(curves), np.isfinite(rates)
    target = np.r_[curves[curve_ok], rate_weight * rates[rate_ok]]

    def profile(log_tau, weights_only=False):
        curve, rate = _design(log_tau, past, hold, features, dt, times, act)
        design = np.concatenate((curve[curve_ok], rate_weight * rate[rate_ok]))
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        return weights if weights_only else design @ weights - target

    best = min((least_squares(profile, np.log(start),
                              bounds=(np.log([.02, .04]), np.log([40., 100.])),
                              diff_step=2e-5, max_nfev=100)
                for start in ((.5, 4.), (.2, 2.), (1., 10.))), key=lambda fit: fit.cost)
    log_tau = best.x
    # The crystallization terms use the fast branch. If the gate is closed,
    # the two Prony branches are interchangeable and are reported fast first.
    if not np.any(act):
        log_tau = np.sort(log_tau)
    return log_tau, profile(log_tau, weights_only=True)


def _fit(past, hold, features, dt, times, act, curves, rates):
    log_tau, weights = _profile_fit(past, hold, features, dt, times, act, curves, rates)
    curve_ok, rate_ok = np.isfinite(curves), np.isfinite(rates)

    def residual(p):
        curve, rate = _design(p[:2], past, hold, features, dt, times, act)
        return ((np.einsum('nkp,p->nk', curve, p[2:]) - curves)[curve_ok],
                (np.einsum('np,p->n', rate, p[2:]) - rates)[rate_ok])

    def mad(r):
        return max(1e-7, 1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else 1.

    start = np.r_[log_tau, weights]
    bounds = (np.r_[np.log([.02, .04]), np.full(9, -np.inf)],
              np.r_[np.log([40., 100.]), np.full(9, np.inf)])
    for _ in range(2):
        scales = [mad(r) for r in residual(start)]
        def normalized(p):
            return np.concatenate([r / s for r, s in zip(residual(p), scales)])
        fit = least_squares(normalized, start, bounds=bounds, loss='soft_l1',
                            f_scale=1.5, x_scale='jac', max_nfev=120)
        start = fit.x
    return start[:2], start[2:]


def solve_alpha_m97_c02_batch_04(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    act = _activation(dataset, past, dt)
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, weights = _fit(past[:n], hold[:n], features[:n], dt, times, act[:n], curves, rates)
    curve, rate = _design(log_tau, past[n:], hold[n:], features[n:], dt, times, act[n:])
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
