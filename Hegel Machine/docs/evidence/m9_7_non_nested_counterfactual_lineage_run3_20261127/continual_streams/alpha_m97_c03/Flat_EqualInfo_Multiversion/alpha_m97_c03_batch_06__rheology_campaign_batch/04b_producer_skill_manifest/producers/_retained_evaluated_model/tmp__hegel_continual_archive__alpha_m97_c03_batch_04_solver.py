"""Load-cell calibrated two-branch model for crystallized creep."""
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


def _basis(logtau, past, hold, features, dt, t, damage, plateau):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    u = dt[:, None] / tau
    for j, (a, b) in enumerate(zip(np.exp(-u), -np.expm1(-u))):
        z = a * z + b * past[:, j, None]
    # The final row is the time derivative at zero, not an extra horizon.
    phi = np.vstack((-np.expm1(-t[:, None] / tau), 1 / tau))
    terminal = np.r_[np.log1p(t) if plateau else t, 1.0]
    act = np.tanh(damage)
    a = np.empty((len(hold), len(t) + 1, 9))
    a[:, :, :2] = (hold[:, None] - z)[:, None, :] * phi
    a[:, :, 2] = terminal
    a[:, :, 3:7] = features[:, None, :] * terminal[None, :, None]
    a[:, :, 7] = (act * hold)[:, None] * phi[:, 0]
    a[:, :, 8] = (act * z[:, 0])[:, None] * terminal
    return a


def _damage(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.5:
        return np.zeros(len(past))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)
    return np.sum(excess**2 * dt, axis=1)


def _scale(e):
    return max(1e-6, 1.4826 * np.median(np.abs(e - np.median(e))))


def _fit(args, target):
    mult = np.ones_like(target)
    bounds = tuple(np.log([0.015, 120.0]))
    starts = np.log(((0.4, 3.0), (1.0, 7.0), (3.0, 0.4)))

    def fit_weighted(starts):
        y = (target * mult).ravel()

        def project(logtau):
            a = (_basis(logtau, *args) * mult[:, :, None]).reshape(-1, 9)
            w = np.linalg.lstsq(a, y, rcond=None)[0]
            return np.einsum('ij,j->i', a, w) - y, w

        best = min((least_squares(lambda p: project(p)[0], start,
                                  bounds=bounds, max_nfev=120) for start in starts),
                   key=lambda result: result.cost)
        return best.x, project(best.x)[1]

    logtau, w = fit_weighted(starts)
    err = _basis(logtau, *args) @ w - target
    # The rate channel and strain trajectories have different noise scales.
    for k in range(12):
        if k in (0, 3, 6, 9):
            scales = np.r_[np.repeat(_scale(err[:, :-1]), target.shape[1] - 1), _scale(err[:, -1])]
        c = 1.5 if k < 3 else 2.385
        mult = 1 / (scales * np.hypot(1, err / (c * scales)))
        logtau, w = fit_weighted(starts)
        err = _basis(logtau, *args) @ w - target
    return logtau, w


def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:
    past, hold, features, offset, dt, t = _prepare(dataset)
    damage = _damage(dataset, past, dt)
    plateau = dataset['shear_banding_index'] <= 0.5
    train = dataset['train']
    n = len(train['lab_id'])
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    logtau, w = _fit((past[:n], hold[:n], features[:n], dt, t, damage[:n], plateau), target)
    pred = _basis(logtau, past[n:], hold[n:], features[n:], dt, t, damage[n:], plateau) @ w
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
