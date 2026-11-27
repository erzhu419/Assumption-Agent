"""Calibrated two-branch Prony model with linear terminal flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    use = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _historical(lab):
    if _ARCHIVE.is_file():
        for sweep in reversed(json.loads(_ARCHIVE.read_text())):
            x, y = _channel(sweep, lab)
            if len(x) >= 2 and np.ptp(x) > 0:
                return _affine(x, y)
    return None


def _calibrate(sweep):
    calibration = np.empty((4, 2))
    for lab in range(4):
        x, y = _channel(sweep, lab)
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3 and np.ptp(x) > 0:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            slopes = (y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]])
            gain = np.median(slopes)
            offset = np.median(y - gain*x)
            clean = np.abs(y - gain*x - offset) <= 3.0
        if len(x) < 2 or (not clean.all() and clean.sum() < 4):
            archived = _historical(lab)
            if archived is not None:
                calibration[lab] = archived
                continue
        if clean.sum() < 2 or np.ptp(x[clean]) == 0:
            raise ValueError(f'No identifiable load-cell calibration for channel {lab}')
        calibration[lab] = _affine(x[clean], y[clean])
    return calibration


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def both(key):
        return np.concatenate((train[key], test[key]))

    labs, strata = both('lab_id'), both('stratum')
    raw, refs = both('coupon_features_raw'), both('ref_coupons')
    x = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        use = labs == lab
        panel = refs[use]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (raw[use] - center) / scale
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = []
    for index in np.unique(cell):
        values = x[cell == index]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(x.shape[1], np.nan), where=counts != 0))
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    gains, biases = _calibrate(dataset['calibration_load_cells'])[labs].T
    past = gains[:, None] * both('past_stress_cmd') + biases[:, None]
    hold = gains * both('stress_hold_cmd') + biases
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _design(past, hold, x, dt, times, taus):
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    survival = np.exp(-later[:, None] / taus)
    gap = hold[:, None] - past @ (-np.expm1(-dt[:, None] / taus) * survival)
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-times[:, None] / taus),
                            times[None, :, None] * x[:, None, :]), axis=2)
    rate = np.column_stack((gap / taus, x))
    return curve, rate


def _fit(past, hold, x, dt, times, curves, rates, rate_weight=1.):
    curve_mask, rate_mask = np.isfinite(curves), np.isfinite(rates)
    target = np.r_[curves[curve_mask], rate_weight * rates[rate_mask]]

    def profile(log_taus):
        curve, rate = _design(past, hold, x, dt, times, np.exp(log_taus))
        matrix = np.vstack((curve[curve_mask], rate_weight * rate[rate_mask]))
        weights = np.linalg.lstsq(matrix, target, rcond=1e-10)[0]
        return np.einsum('ij,j->i', matrix, weights) - target, weights

    best = min((least_squares(lambda q: profile(q)[0], np.log(start),
                              bounds=np.log([0.02, 40.]), max_nfev=120)
                for start in ((.10, 1.), (.30, 2.), (.65, 4.), (1.3, 8.))),
               key=lambda result: result.cost)
    log_taus = np.sort(best.x)
    err, weights = profile(log_taus)
    nc = curve_mask.sum()

    def mad(v):
        return max(1e-6, 1.4826 * np.median(np.abs(v - np.median(v))))

    def raw(p):
        curve, rate = _design(past, hold, x, dt, times, np.exp(p[:2]))
        return np.r_[(curve @ p[2:])[curve_mask] - curves[curve_mask],
                     (rate @ p[2:])[rate_mask] - rates[rate_mask]]

    p = np.r_[log_taus, weights]
    for _ in range(2):
        e = raw(p)
        scale = np.r_[np.full(nc, mad(e[:nc])), np.full(rate_mask.sum(), mad(e[nc:]))]
        p = least_squares(lambda q: raw(q) / scale, p, loss='cauchy', f_scale=2.5,
                          bounds=(np.r_[np.log([.02, .02]), np.full(7, -np.inf)],
                                  np.r_[np.log([40., 40.]), np.full(7, np.inf)]),
                          x_scale='jac', max_nfev=120).x
    if p[0] > p[1]:
        p[:2] = p[:2][::-1]
        p[2:4] = p[2:4][::-1]
    return np.exp(p[:2]), p[2:]


def solve_alpha_m97_c02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(*(a[:n][complete] for a in data), dt, times,
                        curves[complete], train['instant_rate_obs'][complete])
    curve, rate = _design(*(a[n:] for a in data), dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
