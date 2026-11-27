"""Two-branch hereditary strain and logarithmic plateau for campaign batch 07."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sweep(table, channel):
    x, y = (np.asarray(table[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    keep = (np.asarray(table['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _sweep_fit(x, y):
    i, j = np.triu_indices(x.size, 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    if not i.size:
        return None, 0, False
    g = (y[j] - y[i]) / (x[j] - x[i])
    b = y[i] - g * x[i]
    error = np.abs(y - g[:, None] * x - b[:, None])
    consensus = error < 0.025
    best = np.lexsort((np.minimum(error, 0.025).sum(axis=1), -consensus.sum(axis=1)))[0]
    clean = consensus[best]
    emi = bool(np.max(error[best]) > 3.0)
    use = clean if emi else np.ones(x.size, dtype=bool)
    return np.polyfit(x[use], y[use], 1), int(use.sum()), emi


def _calibrate(table):
    archive = None
    calibration = []
    for channel in range(4):
        x, y = _sweep(table, channel)
        fit, count, emi = _sweep_fit(x, y)
        if fit is None or (emi and count < 4):
            if archive is None:
                archive = json.loads(_ARCHIVE.read_text())
            # Use the most recent identifiable sweep; a pair among EMI
            # spikes is not a reliable line, even when it fits exactly.
            for sweep in reversed(archive):
                u, v = _sweep(sweep, channel)
                candidate, clean_count, contaminated = _sweep_fit(u, v)
                if candidate is not None and (not contaminated or clean_count >= 4):
                    fit = candidate
                    break
        if fit is None:
            raise ValueError(f'No usable load-cell sweep for channel {channel}')
        calibration.append(fit)
    return np.array(calibration)


def _align(lab, stratum, raw, ref):
    x = np.array(raw, dtype=float, copy=True)
    for channel in np.unique(lab):
        rows = lab == channel
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).any(axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale

    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        count = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), count, out=means[i], where=count > 0)
    overall = np.nanmean(means, axis=0)
    for group in np.unique(stratum):
        rows = stratum == group
        donors = means[cells[:, 1] == group]
        count = np.isfinite(donors).sum(axis=0)
        fill = np.divide(np.nansum(donors, axis=0), count, out=overall.copy(), where=count > 0)
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.column_stack((np.ones(len(x)), x))


def _prepare(dataset, calibration=None):
    train, test = dataset['train'], dataset['test']
    keys = ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons',
            'past_stress_cmd', 'stress_hold_cmd', 'strain_0')
    lab, stratum, raw, ref, past_cmd, hold_cmd, strain0 = (
        np.concatenate((train[k], test[k])) for k in keys)
    if calibration is None:
        calibration = _calibrate(dataset['calibration_load_cells'])
    gain, bias = calibration[lab].T
    past = gain[:, None] * past_cmd + bias[:, None]
    hold = gain * hold_cmd + bias
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
    damage = (excess**2) @ dt
    plateau = dataset['shear_banding_index'] <= 0.50
    act = np.tanh(damage) if plateau else np.zeros_like(damage)
    return (past, hold, _align(lab, stratum, raw, ref), act), strain0, dt, times, plateau


def _design(log_tau, inputs, dt, times, plateau):
    past, hold, features, act = inputs
    tau = np.exp(log_tau)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    z = past @ (np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau))
    gap = hold[:, None] - z
    step = -np.expm1(-times[:, None] / tau)
    phi = np.log1p(times) if plateau else times
    ah, az = act * hold, act * z[:, 0]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            ah[:, None, None] * step[None, :, :1],
                            az[:, None, None] * phi[None, :, None]), axis=2)
    rate = np.column_stack((gap / tau, features, ah / tau[0], az))
    return curve, rate


def _fit(inputs, dt, times, plateau, curves, rates):
    yc, yr = np.asarray(curves), np.asarray(rates)
    mc, mr = np.isfinite(yc), np.isfinite(yr)
    y = np.r_[yc[mc], yr[mr]]
    nc = mc.sum()
    row_weight = np.r_[np.ones(nc), np.full(mr.sum(), 2.)]
    yw = y * row_weight

    def matrix(t):
        curve, rate = _design(t, inputs, dt, times, plateau)
        return np.concatenate((curve[mc], rate[mr]), axis=0)

    def project(t, return_params=False):
        a = matrix(t) * row_weight[:, None]
        w = lstsq(a, yw, lapack_driver='gelsy', check_finite=False)[0]
        r = np.einsum('ij,j->i', a, w) - yw
        return (r @ r, np.r_[t, w]) if return_params else r

    lo, hi = np.log([0.025, 80.])
    starts = ((0.25, 2.5), (0.8, 7.), (2.5, 0.25), (7., 0.8))
    trials = [project(least_squares(project, np.log(t), bounds=(lo, hi),
                                   max_nfev=55).x, True) for t in starts]
    trials.sort(key=lambda fit: fit[0])
    limits = (np.r_[lo, lo, np.full(9, -np.inf)], np.r_[hi, hi, np.full(9, np.inf)])

    def residual(p):
        return np.einsum('ij,j->i', matrix(p[:2]), p[2:]) - y

    def mad(r):
        return max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else 1.

    def scales(p):
        r = residual(p)
        return np.r_[np.full(nc, mad(r[:nc])), np.full(mr.sum(), mad(r[nc:]))]

    scale = scales(trials[0][1])
    soft = [least_squares(lambda p: residual(p) / scale, start, bounds=limits,
                          loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=85)
            for _, start in trials[:3]]
    p = min(soft, key=lambda f: f.cost).x
    # Suppress isolated high-amplitude strain readout errors after finding
    # the constitutive fit with the broader basin of the soft-L1 objective.
    scale = scales(p)
    fit = least_squares(lambda q: residual(q) / scale, p, bounds=limits,
                        loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=70)
    return fit.x[:2], fit.x[2:]


def solve_alpha_m97_c02_batch_07(dataset: dict) -> dict:
    inputs, strain0, dt, times, plateau = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, w = _fit(tuple(a[:n] for a in inputs), dt, times, plateau, curves, rates)
    curve, rate = _design(log_tau, tuple(a[n:] for a in inputs), dt, times, plateau)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': strain0[n:, None] + curve @ w,
            'initial_rate_pred': rate @ w}
