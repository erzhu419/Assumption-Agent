"""Calibrated two-branch Prony regression with a damage-gated plateau."""
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
    # Prefer the most recent prefix sweep with an identifiable channel fit.
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        take = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[take]
        y = np.asarray(sweep['true_stress'])[take]
        if len(x) >= 2 and np.ptp(x) > 0:
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


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    aligned = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    past, hold = (np.concatenate([a, b]) for a, b in
                  zip(_physical(train, calibration), _physical(test, calibration)))
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    damage = np.square(np.maximum(np.abs(past) - dataset['crit_stress'], 0.)) @ dt
    # The low-shear phase uses logarithmic flow even when damage is zero.
    # Damage gates the corrections, not the terminal time basis.
    logarithmic = np.broadcast_to(np.asarray(dataset['shear_banding_index']) <= .50, damage.shape)
    active = np.where(logarithmic, np.tanh(damage), 0.)
    return past, hold, aligned, logarithmic, active, dt, times


def _design(past, hold, aligned, logarithmic, active, dt, times, taus):
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = past @ (-np.expm1(-dt[:, None] / taus) * np.exp(-later[:, None] / taus))
    gap = hold[:, None] - memory
    response = -np.expm1(-times[:, None] / taus)
    terminal = np.where(logarithmic[:, None], np.log1p(times), times)
    x = np.column_stack((np.ones(len(hold)), aligned))
    extra = np.column_stack((active * hold, active * memory[:, 0]))
    curve = np.concatenate((gap[:, None, :] * response,
                            x[:, None, :] * terminal[:, :, None],
                            extra[:, None, :1] * response[None, :, :1],
                            extra[:, None, 1:] * terminal[:, :, None]), axis=2)
    rate = np.column_stack((gap / taus, x, extra[:, 0] / taus[0], extra[:, 1]))
    return curve, rate


def _fit(data, curves, rates):
    *_, dt, times = data
    target = np.r_[curves.ravel(), rates]
    bounds = (np.full(2, np.log(dt.min() / 30)),
              np.full(2, np.log(times.max() * 30)))

    def design(log_tau):
        c, r = _design(*data, np.exp(log_tau))
        return np.vstack((c.reshape(-1, 9), r))

    def profile(log_tau, inv_scale):
        a = design(log_tau) * inv_scale[:, None]
        y = target * inv_scale
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.einsum('ij,j->i', a, w) - y, w

    unit = np.ones(len(target))
    starts = np.log([[.15, 1.0], [.4, 2.5], [.9, 5.]])
    best = None
    for start in starts:
        start = np.clip(start, *bounds)
        fit = least_squares(lambda q: profile(q, unit)[0], start, bounds=bounds,
                            max_nfev=100)
        if best is None or fit.cost < best.cost:
            best = fit
    err, weights = profile(best.x, unit)
    taus = np.exp(best.x)
    if taus[0] > taus[1]:
        taus = taus[::-1]
        weights = profile(np.log(taus), unit)[1]
    # Downweight transient observation spikes without discarding the rest of a curve.
    err = profile(np.log(taus), unit)[0]
    nc = curves.size
    mad = lambda v: max(1e-6, 1.4826 * np.median(np.abs(v - np.median(v))))
    scales = np.r_[np.full(nc, mad(err[:nc])), np.full(len(rates), mad(err[nc:]))]
    lo, hi = bounds

    def residual(p):
        return (np.einsum('ij,j->i', design(p[:2]), p[2:]) - target) / scales

    robust = least_squares(residual, np.r_[np.log(taus), weights],
                           bounds=(np.r_[lo, np.full(9, -np.inf)],
                                   np.r_[hi, np.full(9, np.inf)]),
                           loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    taus = np.exp(robust.x[:2])
    if taus[0] <= taus[1]:
        return taus, robust.x[2:]
    # The damage terms reference the fast branch; re-estimate their coefficients.
    taus = taus[::-1]
    return taus, profile(np.log(taus), 1. / scales)[1]


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    # Estimate material parameters from observed features; imputation is for prediction.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    taus, w = _fit(tuple(a[:n][complete] for a in data[:5]) + data[5:],
                   (train['strain_curves_obs'] - train['strain_0'][:, None])[complete],
                   train['instant_rate_obs'][complete])
    curve, rate = _design(*(tuple(a[n:] for a in data[:5]) + data[5:]), taus)
    return {'estimated_taus': taus, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ w,
            'initial_rate_pred': rate @ w}
