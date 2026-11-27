"""Calibrated two-branch Prony regression for amorphous terminal flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _historical_calibration(lab):
    if not _ARCHIVE.is_file():
        return None
    data = json.loads(_ARCHIVE.read_text())
    if isinstance(data, dict):
        data = data.get('calibration_load_cells', data.get('channels', data))
    if isinstance(data, dict) and 'lab_id' in data:
        ids = np.asarray(data['lab_id'])
        x = np.asarray(data['cmd_stress'])[ids == lab]
        y = np.asarray(data['true_stress'])[ids == lab]
        return _affine(x, y) if len(x) >= 2 and np.ptp(x) > 0 else None
    if isinstance(data, dict):
        row = data.get(str(lab))
        data = [row] if isinstance(row, dict) else row
    if isinstance(data, list):
        rows = [r for r in data if r.get('lab_id', lab) == lab]
        fits = [(r['gain'], r['bias']) for r in rows if 'gain' in r and 'bias' in r]
        if fits:
            return np.median(fits, axis=0)
        points = [(r['cmd_stress'], r['true_stress']) for r in rows
                  if 'cmd_stress' in r and 'true_stress' in r]
        if len(points) >= 2:
            x, y = np.asarray(points).T
            if np.ptp(x) > 0:
                return _affine(x, y)
    return None


def _calibrate(cal):
    out = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        take = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][take], cal['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        spike = False
        if len(x) >= 3 and np.ptp(x) > 0:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            slopes = (y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]])
            g = np.median(slopes)
            b = np.median(y - g*x)
            clean = np.abs(y - (g*x + b)) <= 3.
            spike = not clean.all()
        identifiable = clean.sum() >= 2 and np.ptp(x[clean]) > 0
        if identifiable:
            out[lab] = _affine(x[clean], y[clean])
        if len(x) < 2 or (spike and clean.sum() < 4):
            fallback = _historical_calibration(lab)
            if fallback is not None:
                out[lab] = fallback
            elif len(x) == 1:
                out[lab, 1] = y[0] - x[0]
    return out


def _align(train, test):
    labs, strata, raw, refs = (np.concatenate((train[k], test[k])) for k in
                               ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        take = labs == lab
        ref = refs[take]
        mean = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0))
        aligned[take] = (raw[take] - mean) / scale
    _, cells = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    cell_means = []
    for cell in np.unique(cells):
        values = aligned[cells.ravel() == cell]
        counts = np.isfinite(values).sum(axis=0)
        cell_means.append(np.divide(np.nansum(values, axis=0), counts,
                                    out=np.full(raw.shape[1], np.nan), where=counts > 0))
    fill = np.nanmean(cell_means, axis=0)
    return np.where(np.isfinite(aligned), aligned, fill)


def _physical(split, calibration):
    g, b = calibration[split['lab_id']].T
    return g[:, None] * split['past_stress_cmd'] + b[:, None], g * split['stress_hold_cmd'] + b


def _basis(past, hold, dt, times, taus):
    # Unroll the recurrence: a contribution at s decays over all later steps.
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory_weights = -np.expm1(-dt[:, None]/taus) * np.exp(-later[:, None]/taus)
    gap = hold[:, None] - past @ memory_weights
    response = -np.expm1(-times[:, None]/taus)
    return gap[:, None, :] * response, gap/taus


def _fit(past, hold, x, dt, times, curves, rates):
    n, k = curves.shape
    drift = np.column_stack((np.ones(n), x))
    tdrift = (drift[:, None, :] * times[None, :, None]).reshape(n*k, 5)
    target = np.r_[curves.ravel(), rates]
    bounds = (np.r_[np.full(2, np.log(dt.min()/30)), np.full(7, -np.inf)],
              np.r_[np.full(2, np.log(times.max()*30)), np.full(7, np.inf)])

    def design(logtau):
        a, b = _basis(past, hold, dt, times, np.exp(logtau))
        return np.vstack((np.column_stack((a.reshape(n*k, 2), tdrift)),
                          np.column_stack((b, drift))))

    def residual(p):
        return np.einsum('ij,j->i', design(p[:2]), p[2:]) - target

    def jac(p):
        h = 1e-4
        d = design(p[:2])
        derivative = []
        for step in ([h, 0.], [0., h]):
            diff = (design(p[:2] + step)[:, :2] - design(p[:2] - step)[:, :2]) / (2*h)
            derivative.append(np.einsum('ij,j->i', diff, p[2:4]))
        return np.column_stack((*derivative, d))

    def fit(p, normalizer, loss):
        return least_squares(lambda q: residual(q)/normalizer, p,
                             jac=lambda q: jac(q)/normalizer[:, None], bounds=bounds,
                             loss=loss, x_scale='jac', max_nfev=120)

    starts = np.log([[0.15, 1.0], [0.4, 2.5], [0.9, 5.]])
    scale = np.r_[np.full(n*k, .05), np.full(n, .05)]
    trials = []
    for start in starts:
        d = design(start)
        w = np.linalg.lstsq(d, target, rcond=None)[0]
        trials.append(fit(np.r_[start, w], scale, 'soft_l1'))
    best = min(trials, key=lambda r: r.cost)
    err = residual(best.x)
    mad = lambda e: 1.4826*np.median(np.abs(e - np.median(e)))
    scale = np.r_[np.full(n*k, max(mad(err[:n*k]), 1.e-6)),
                  np.full(n, max(mad(err[n*k:]), 1.e-6))]
    result = fit(best.x, 2*scale, 'cauchy').x
    order = np.argsort(result[:2])
    return np.exp(result[:2][order]), np.r_[result[2:4][order], result[4:]]


def solve_alpha_m97_c02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    calibration = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    train_past, train_hold = _physical(train, calibration)
    taus, weights = _fit(train_past, train_hold, aligned[:n], dt, times,
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    a, b = _basis(*_physical(test, calibration), dt, times, taus)
    drift = weights[2] + aligned[n:] @ weights[3:]
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + a @ weights[:2] + drift[:, None]*times,
            'initial_rate_pred': b @ weights[:2] + drift}
