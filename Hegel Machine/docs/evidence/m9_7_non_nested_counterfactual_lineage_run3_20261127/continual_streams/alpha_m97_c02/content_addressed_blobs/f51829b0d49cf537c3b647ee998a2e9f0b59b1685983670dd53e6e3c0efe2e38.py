"""Load-cell calibrated, two-branch jammed-slip rheology."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], float)
    y = np.asarray(sweep['true_stress'], float)
    use = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _unspiked(x, y):
    if len(x) < 3:
        return np.ones(len(x), bool)
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    residual = np.abs((y[None, :] - y[i, None]) - slopes[:, None] * (x[None, :] - x[i, None]))
    # Among pair fits, prefer the largest consensus, then the smallest inlier error.
    inlier = residual <= 3.0
    keep = np.lexsort((np.sum(np.where(inlier, residual**2, 0), axis=1), -inlier.sum(axis=1)))[0]
    return inlier[keep]


def _calibrate(sweep):
    coefficients = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _channel(sweep, lab)
        clean = _unspiked(x, y)
        if clean.sum() >= 4 or (len(x) >= 2 and clean.all()):
            coefficients[lab] = _line(x[clean], y[clean])
            continue
        if history is None:
            history = json.loads(_ARCHIVE.read_text())
        for archived in reversed(history):
            ax, ay = _channel(archived, lab)
            good = _unspiked(ax, ay)
            if good.sum() >= 2 and np.ptp(ax[good]) > 0:
                coefficients[lab] = _line(ax[good], ay[good])
                break
        else:
            raise ValueError(f'No calibration sweep for load cell {lab}')
    return coefficients


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, stratum = join('lab_id'), join('stratum')
    refs, raw = join('ref_coupons'), join('coupon_features_raw')
    aligned = np.empty_like(raw, dtype=float)
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        aligned[use] = (raw[use] - center) / scale
    cells = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        values = aligned[(lab == c) & (stratum == s)]
        count = np.isfinite(values).sum(axis=0)
        cells.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(raw.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(cells, axis=0))
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None] * join('past_stress_cmd') + bias[:, None]
    hold = gain * join('stress_hold_cmd') + bias
    terminal = np.column_stack((np.ones_like(hold), aligned,
                                np.sign(hold) * np.sqrt(np.abs(hold))))
    return past, hold, terminal


def _design(past, hold, terminal, dt, times, tau):
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = past @ (-np.expm1(-dt[:, None] / tau) * np.exp(-remaining[:, None] / tau))
    gap = hold[:, None] - memory
    contrast = gap[:, 0] - gap[:, 1]
    relax = -np.expm1(-times[:, None] / tau)
    phi = times / (1 + 0.85 * times)
    curve = np.concatenate((gap[:, None, :] * relax,
                            terminal[:, None, :] * phi[None, :, None],
                            (contrast[:, None] * relax[None, :, 0])[:, :, None]), axis=2)
    rate = np.column_stack((gap / tau, terminal, contrast / tau[0]))
    return curve, rate


def _fit(past, hold, terminal, dt, times, curves, rates):
    nc = curves.size
    target = np.r_[curves.ravel(), rates]

    def design(logtau):
        a, b = _design(past, hold, terminal, dt, times, np.exp(logtau))
        return np.concatenate((a.reshape(-1, a.shape[-1]), b))

    def profile(logtau):
        matrix = design(logtau)
        weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
        return matrix @ weights - target

    bounds = (np.log(0.015), np.log(80.0))
    starts = np.log([[0.23, 2.4], [0.6, 4.5], [1.2, 8.0], [2.4, 0.23], [4.5, 0.6]])
    fits = [least_squares(profile, start, bounds=bounds, max_nfev=100) for start in starts]

    def scales(residual):
        blocks = (residual[:nc], residual[nc:])
        sigma = [max(1e-9, 1.4826 * np.median(np.abs(block - np.median(block))))
                 for block in blocks]
        return np.repeat(sigma, [nc, len(rates)])

    def residual(params, scale):
        return (design(params[:2]) @ params[2:] - target) / scale

    joint_bounds = (np.r_[np.repeat(bounds[0], 2), np.full(9, -np.inf)],
                    np.r_[np.repeat(bounds[1], 2), np.full(9, np.inf)])
    scale = scales(min(fits, key=lambda fit: fit.cost).fun)
    robust = []
    for fit in fits:
        matrix = design(fit.x)
        weights = np.linalg.lstsq(matrix / scale[:, None], target / scale, rcond=None)[0]
        robust.append(least_squares(residual, np.r_[fit.x, weights], args=(scale,),
                                   bounds=joint_bounds, loss='cauchy', f_scale=2.0,
                                   x_scale='jac', max_nfev=100))
    best = min(robust, key=lambda fit: fit.cost)
    scale = scales(best.fun * scale)
    best = least_squares(residual, best.x, args=(scale,), bounds=joint_bounds,
                         loss='cauchy', f_scale=2.0, x_scale='jac', max_nfev=100)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c02_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    past, hold, terminal = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    curves = train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None]
    tau, weights = _fit(past[:n], hold[:n], terminal[:n], dt, times, curves, train['instant_rate_obs'])
    a, b = _design(past[n:], hold[n:], terminal[n:], dt, times, tau)
    return {'estimated_taus': tau, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
