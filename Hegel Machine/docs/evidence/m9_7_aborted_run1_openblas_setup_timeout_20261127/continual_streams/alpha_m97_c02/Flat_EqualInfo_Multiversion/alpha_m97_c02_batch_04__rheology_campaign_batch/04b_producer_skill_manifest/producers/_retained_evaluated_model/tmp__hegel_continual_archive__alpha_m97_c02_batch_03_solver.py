"""Calibrated two-branch creep model for an amorphous melt."""
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


def _basis(log_tau, past, hold, features, age, dt, times, derivatives=False):
    tau = np.exp(log_tau)
    a, q, r = age[:, None]/tau, dt[:, None]/tau, times[:, None]/tau
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    uptake = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ uptake
    step = -np.expm1(-r)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    phi = times/(1 + 0.85*times)
    delta = gap[:, 0] - gap[:, 1]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            slip[:, None, None] * phi[None, :, None],
                            delta[:, None, None] * step[None, :, :1]), axis=2)
    rate = np.column_stack((gap/tau, features, slip, delta/tau[0]))
    if not derivatives:
        return curve, rate
    dg = -(past @ (a*uptake - q*ea*eq))
    ds = -r*er
    dc = dg[:, None, :] * step + gap[:, None, :] * ds
    dr = (dg - gap)/tau
    xc = np.stack((dg[:, :1]*step[:, 0] + delta[:, None]*ds[:, 0],
                   -dg[:, 1:]*step[:, 0]), axis=-1)
    xr = np.column_stack((dg[:, 0]-delta, -dg[:, 1]))/tau[0]
    return curve, rate, np.vstack((dc.reshape(-1, 2), dr)), np.vstack((xc.reshape(-1, 2), xr))


def _fit(past, hold, features, age, dt, times, curves, rates):
    target = np.r_[curves.ravel(), rates]
    observed = np.isfinite(target)
    target = target[observed]
    args = past, hold, features, age, dt, times

    def design(log_tau, derivatives=False):
        curve, rate, *jac = _basis(log_tau, *args, derivatives=derivatives)
        mat = np.vstack((curve.reshape(-1, 9), rate))[observed]
        if derivatives:
            return mat, jac[0][observed], jac[1][observed]
        return mat

    def residual(params):
        return np.einsum('ij,j->i', design(params[:2]), params[2:]) - target

    def jacobian(params):
        mat, branch, cross = design(params[:2], derivatives=True)
        return np.column_stack((branch*params[2:4] + cross*params[-1], mat))

    bounds = (np.r_[np.log([0.01, 0.01]), np.full(9, -np.inf)],
              np.r_[np.log([100., 100.]), np.full(9, np.inf)])
    best = None
    # The cross term acts on branch zero, so exchanging branches is not a symmetry.
    for short, long in ((0.45, 2.4), (0.18, 1.2), (0.20, 6.0), (1., 6.), (0.45, 15.)):
        for pair in ((short, long), (long, short)):
            log_tau = np.log(pair)
            matrix = design(log_tau)
            weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
            for _ in range(3):
                e = np.einsum('ij,j->i', matrix, weights) - target
                w = 1/np.sqrt(1 + (e/0.025)**2)
                weights = np.linalg.lstsq(matrix*w[:, None], target*w, rcond=None)[0]
            fit = least_squares(residual, np.r_[log_tau, weights], jac=jacobian,
                                bounds=bounds, loss='cauchy', f_scale=0.025,
                                x_scale='jac', max_nfev=130)
            if best is None or fit.cost < best.cost:
                best = fit
    return best.x[:2], best.x[2:]


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, weights = _fit(past[:n], hold[:n], features[:n], age, dt, times, curves, rates)
    curve, rate = _basis(log_tau, past[n:], hold[n:], features[n:], age, dt, times)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
