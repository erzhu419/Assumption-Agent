"""Fit a calibrated two-branch creep model to a rheology campaign."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _prior_calibration(lab):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    # The prefix is chronological; use the most recent usable sweep.
    for sweep in reversed(history):
        rows = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[rows]
        y = np.asarray(sweep['true_stress'])[rows]
        clean = np.isfinite(x) & np.isfinite(y)
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            return np.polyfit(x[clean], y[clean], 1)
    raise ValueError(f'No calibration sweep for lab {lab}')


def _calibration(table):
    fits = []
    for lab in range(4):
        rows = (table['lab_id'] == lab) & np.isfinite(table['cmd_stress']) & np.isfinite(table['true_stress'])
        x = table['cmd_stress'][rows]
        y = table['true_stress'][rows]
        if len(x) < 2:
            fits.append(_prior_calibration(lab))
            continue
        fit = np.polyfit(x, y, 1)
        if np.max(np.abs(y - np.polyval(fit, x))) > 0.01:
            # Find the tight affine consensus before testing the 3-unit EMI threshold.
            candidates = []
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    if x[i] == x[j]:
                        continue
                    gain = (y[j] - y[i]) / (x[j] - x[i])
                    bias = y[i] - gain * x[i]
                    e = np.abs(y - (gain * x + bias))
                    clean = e <= 0.01
                    if np.any(e > 3.0) and np.all(clean | (e > 3.0)):
                        candidates.append(clean)
            if candidates:
                largest = max(mask.sum() for mask in candidates)
                masks = np.unique([mask for mask in candidates if mask.sum() == largest], axis=0)
                if largest >= 4 or len(masks) == 1:
                    clean = masks[0]
                else:
                    prior = _prior_calibration(lab)
                    e = np.abs(y - np.polyval(prior, x))
                    clean = min(masks, key=lambda mask: np.mean(e[mask] ** 2))
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
    for c in np.unique(lab):
        rows = lab == c
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
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0) ** 2 @ dt
    active = (dataset['shear_banding_index'] <= 0.5) & (damage > 0)
    act = np.where(active, np.tanh(damage), 0.0)
    return past, hold, features, act, active, join('strain_0'), dt, t


def _basis(logtaus, past, hold, features, act, active, dt, t, plateau='gated'):
    tau = np.exp(logtaus)
    u, v = dt[:, None] / tau, t[:, None] / tau
    decay, relax = np.exp(-u), -np.expm1(-v)
    drelax = -v * np.exp(-v)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for step, (a, da) in enumerate(zip(decay, u * decay)):
        stress = past[:, step, None]
        dz = a * dz + da * (z - stress)
        z = a * z + (-np.expm1(-u[step])) * stress
    gap = hold[:, None] - z
    ramp = np.broadcast_to(t, (len(hold), len(t)))
    if plateau == 'all':
        ramp = np.broadcast_to(np.log1p(t), ramp.shape)
    elif plateau == 'gated':
        ramp = np.where(active[:, None], np.log1p(t), ramp)
    ramp = np.column_stack((ramp, np.ones(len(hold))))
    relax = np.vstack((relax, 1 / tau))
    drelax = np.vstack((drelax, -1 / tau))
    design = np.empty((len(hold), len(t) + 1, 9))
    design[:, :, :2] = gap[:, None, :] * relax
    design[:, :, 2:7] = ramp[:, :, None] * np.column_stack((np.ones(len(hold)), features))[:, None, :]
    design[:, :, 7] = (act * hold)[:, None] * relax[:, 0]
    design[:, :, 8] = (act * z[:, 0])[:, None] * ramp
    dhered = gap[:, None, :] * drelax - dz[:, None, :] * relax
    dreg0 = (act * hold)[:, None] * drelax[:, 0]
    dreg1 = (act * dz[:, 0])[:, None] * ramp
    return design, dhered, dreg0, dreg1


def _scale(error):
    return max(1.4826 * np.median(np.abs(error - np.median(error))), 1e-7)


def _fit(args, target, plateau='gated'):
    def model(p, jac=False):
        a, dh, d0, d1 = _basis(p[:2], *args, plateau)
        a = a.reshape(-1, 9)
        if not jac:
            return np.einsum('ij,j->i', a, p[2:]) - target
        h = dh.reshape(-1, 2) * p[2:4]
        h[:, 0] += d0.ravel() * p[9] + d1.ravel() * p[10]
        return np.column_stack((h, a))

    lower = np.r_[np.log(0.015), np.log(0.015), np.full(9, -np.inf)]
    upper = np.r_[np.log(120.0), np.log(120.0), np.full(9, np.inf)]
    starts = []
    for pair in ((0.25, 2.0), (0.65, 4.5), (1.5, 8.0), (3.0, 0.45), (7.0, 1.0)):
        logs = np.log(pair)
        a = _basis(logs, *args, plateau)[0].reshape(-1, 9)
        starts.append(np.r_[logs, np.linalg.lstsq(a, target, rcond=None)[0]])

    def fit(start, loss, scale):
        return least_squares(model, start, jac=lambda p: model(p, True), bounds=(lower, upper),
                             loss=loss, f_scale=scale, x_scale='jac', max_nfev=160)

    scale = min(_scale(model(p)) for p in starts)
    best = min((fit(p, 'soft_l1', scale) for p in starts), key=lambda result: result.cost)
    for _ in range(2):
        best = fit(best.x, 'cauchy', 2.0 * _scale(best.fun))
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    past, hold, features, act, active, offset, dt, t = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    grouped = (past, hold, features, act, active)
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    plateau = 'all' if dataset['shear_banding_index'] <= 0.5 else 'linear'
    taus, weights = _fit((*[x[:n] for x in grouped], dt, t), target.ravel(), plateau)
    basis = _basis(np.log(taus), *[x[n:] for x in grouped], dt, t, plateau)[0]
    pred = basis @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
