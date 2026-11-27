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
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(log_taus, past, hold, features, age, dt, times, derivatives=False):
    tau = np.exp(log_taus)
    a, q, r = age[:, None]/tau, dt[:, None]/tau, times[:, None]/tau
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    # A history step contributes its uptake times its decay over subsequent steps.
    memory = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ memory
    step = -np.expm1(-r)
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap/tau, features))
    if not derivatives:
        return curve, rate
    dz = past @ (a*memory - q*ea*eq)
    dc = -dz[:, None, :]*step - gap[:, None, :]*r*er
    dr = (-dz - gap)/tau
    return curve, rate, dc, dr


def _fit(past, hold, features, age, dt, times, curves, rates):
    target = np.r_[curves.ravel(), rates]
    observed = np.isfinite(target)
    target = target[observed]
    args = past, hold, features, age, dt, times

    def design(log_taus, derivatives=False):
        c, r, *jac = _basis(log_taus, *args, derivatives=derivatives)
        matrix = np.vstack((c.reshape(-1, 7), r))[observed]
        if derivatives:
            dc, dr = jac
            return matrix, np.vstack((dc.reshape(-1, 2), dr))[observed]
        return matrix

    def residual(params):
        return np.einsum('ij,j->i', design(params[:2]), params[2:]) - target

    def jacobian(params):
        matrix, diff = design(params[:2], derivatives=True)
        return np.column_stack((diff * params[2:4], matrix))

    lo = np.r_[np.log([0.01, 0.01]), np.full(7, -np.inf)]
    hi = np.r_[np.log([100., 100.]), np.full(7, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0),
                 (1.0, 6.0), (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        weights = np.linalg.lstsq(design(log_taus), target, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, weights], jac=jacobian,
                            bounds=(lo, hi), loss='cauchy', f_scale=0.025,
                            x_scale='jac', max_nfev=130)
        if best is None or fit.cost < best.cost:
            best = fit
    params = best.x.copy()
    if params[0] > params[1]:
        params[[0, 1, 2, 3]] = params[[1, 0, 3, 2]]
    return params[:2], params[2:]


def solve_alpha_m97_c02_batch_02(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_taus, weights = _fit(past[:n], hold[:n], features[:n], age, dt, times, curves, rates)
    curve, rate = _basis(log_taus, past[n:], hold[n:], features[n:], age, dt, times)
    return {'estimated_taus': np.exp(log_taus), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
