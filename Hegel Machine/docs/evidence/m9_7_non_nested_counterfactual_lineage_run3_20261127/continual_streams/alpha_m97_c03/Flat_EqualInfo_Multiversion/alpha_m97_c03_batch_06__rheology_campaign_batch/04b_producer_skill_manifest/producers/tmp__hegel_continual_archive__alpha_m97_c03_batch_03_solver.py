"""Calibrated two-branch creep model for the jammed-slip campaign."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _prior_calibration(lab):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    for sweep in reversed(history):
        rows = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[rows]
        y = np.asarray(sweep['true_stress'])[rows]
        clean = np.isfinite(x) & np.isfinite(y)
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            return np.polyfit(x[clean], y[clean], 1)
    raise ValueError(f'No calibration sweep for channel {lab}')


def _calibration(table):
    lab_id, cmd, true = (np.asarray(table[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    fits = []
    for lab in range(4):
        rows = (lab_id == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[rows], true[rows]
        if len(x) < 2:
            fits.append(_prior_calibration(lab))
            continue
        fit = np.polyfit(x, y, 1)
        if np.max(np.abs(y - np.polyval(fit, x))) > 0.01:
            # Find the largest tight affine consensus before identifying EMI.
            masks = []
            for i, j in zip(*np.triu_indices(len(x), 1)):
                if x[i] == x[j]:
                    continue
                gain = (y[j] - y[i]) / (x[j] - x[i])
                error = np.abs((y - y[i]) - gain * (x - x[i]))
                clean = error <= 0.01
                if np.any(error > 3.0) and np.all(clean | (error > 3.0)):
                    masks.append(clean)
            if masks:
                largest = max(mask.sum() for mask in masks)
                options = np.unique([mask for mask in masks if mask.sum() == largest], axis=0)
                clean = options[0]
                if largest < 4 and len(options) > 1:
                    prior = _prior_calibration(lab)
                    error = (y - np.polyval(prior, x)) ** 2
                    clean = min(options, key=lambda mask: error[mask].mean())
                fit = np.polyfit(x[clean], y[clean], 1)
        fits.append(fit)
    return np.asarray(fits)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]))

    lab, stratum = join('lab_id'), join('stratum')
    ref, features = join('ref_coupons'), join('coupon_features_raw').astype(float)
    cells = []
    for channel in np.unique(lab):
        rows = lab == channel
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        features[rows] = (features[rows] - center) / scale
        for s in np.unique(stratum[rows]):
            cell = features[rows & (stratum == s)]
            count = np.isfinite(cell).sum(axis=0)
            cells.append(np.divide(np.nansum(cell, axis=0), count,
                                   out=np.full(features.shape[1], np.nan), where=count > 0))
    features = np.where(np.isnan(features), np.nanmean(cells, axis=0), features)
    gain, bias = _calibration(dataset['calibration_load_cells'])[lab].T
    past = join('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = join('stress_hold_cmd') * gain + bias
    return past, hold, features, join('strain_0'), np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])


def _basis(logtau, past, hold, features, dt, t):
    tau = np.exp(logtau)
    u, v = dt[:, None] / tau, t[:, None] / tau
    decay = np.exp(-u)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for step, (a, b, da) in enumerate(zip(decay, -np.expm1(-u), u * decay)):
        stress = past[:, step, None]
        dz = a * dz + da * (z - stress)
        z = a * z + b * stress
    gap = hold[:, None] - z
    phi = np.vstack((-np.expm1(-v), 1 / tau))
    dphi = np.vstack((-v * np.exp(-v), -1 / tau))
    terminal = np.r_[t / (1 + 0.85 * t), 1.0]
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    a = np.empty((len(hold), len(t) + 1, 9))
    a[:, :, :2] = gap[:, None, :] * phi
    a[:, :, 2:8] = np.column_stack((np.ones(len(hold)), features, slip))[:, None, :] * terminal[None, :, None]
    delta = gap[:, :1] - gap[:, 1:]
    a[:, :, 8] = delta * phi[:, 0]
    da = gap[:, None, :] * dphi - dz[:, None, :] * phi
    ds = np.stack((delta * dphi[:, 0] - dz[:, :1] * phi[:, 0], dz[:, 1:] * phi[:, 0]), axis=-1)
    return a, da, ds


def _scale(e):
    return max(1.4826 * np.median(np.abs(e - np.median(e))), 1e-7)


def _fit(args, target, mode='cauchy'):
    y = target.ravel()
    mult = np.ones_like(target)
    cached = None

    def terms(logs):
        nonlocal cached, a, da, ds
        if cached is None or not np.array_equal(logs, cached):
            basis, derivative, cross = _basis(logs, *args)
            a, da, ds = basis.reshape(-1, 9), derivative.reshape(-1, 2), cross.reshape(-1, 2)
            cached = logs.copy()
        return a, da, ds

    a = da = ds = None

    def residual(p):
        basis, _, _ = terms(p[:2])
        return np.einsum('ij,j->i', basis, p[2:]) - y

    def fun(p):
        return residual(p) * mult.ravel()

    def jac(p):
        basis, derivative, cross = terms(p[:2])
        return np.column_stack((derivative * p[2:4] + cross * p[-1], basis)) * mult.reshape(-1, 1)

    bounds = (np.r_[np.log([0.015, 0.015]), [-np.inf] * 9],
              np.r_[np.log([120., 120.]), [np.inf] * 9])

    def fit(p, loss, scale):
        return least_squares(fun, p, jac=jac, bounds=bounds, loss=loss,
                             f_scale=scale, x_scale='jac', max_nfev=140)

    starts = []
    for pair in ((0.3, 2.0), (0.65, 4.5), (1.5, 8.0), (2.0, 0.3), (4.5, 0.65), (8.0, 1.5)):
        logs = np.log(pair)
        basis, _, _ = terms(logs)
        starts.append(np.r_[logs, np.linalg.lstsq(basis, y, rcond=None)[0]])
    scale = min(_scale(residual(p)) for p in starts)
    best = min((fit(p, 'soft_l1', scale) for p in starts), key=lambda result: result.cost)
    if mode != 'soft_l1':
        for _ in range(3):
            err = residual(best.x).reshape(target.shape)
            curve_scale, rate_scale = _scale(err[:, :-1]), _scale(err[:, -1])
            mult[:, -1] = curve_scale / rate_scale
            best = fit(best.x, mode, 2.25 * curve_scale)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    past, hold, features, offset, dt, t = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    taus, weights = _fit((past[:n], hold[:n], features[:n], dt, t), target)
    basis, _, _ = _basis(np.log(taus), past[n:], hold[n:], features[n:], dt, t)
    pred = basis @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
