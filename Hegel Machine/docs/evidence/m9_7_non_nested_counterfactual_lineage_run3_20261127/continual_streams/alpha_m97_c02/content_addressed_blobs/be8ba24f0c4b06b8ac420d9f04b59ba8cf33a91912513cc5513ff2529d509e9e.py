"""Two-branch hereditary creep with saturated jammed slip."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _points(sweep, lab):
    ids, x, y = (np.asarray(sweep[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    take = (ids == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _clean(x, y):
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    if len(x) < 3 or not distinct.any():
        return np.ones(len(x), dtype=bool)
    i, j = i[distinct], j[distinct]
    gain = np.median((y[j] - y[i]) / (x[j] - x[i]))
    intercepts = y - gain * x
    return np.abs(intercepts - np.median(intercepts)) <= 3.0


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibrate(sweep):
    out = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _points(sweep, lab)
        clean = _clean(x, y)
        if clean.sum() >= 4 or (len(x) >= 2 and clean.all()):
            out[lab] = _affine(x[clean], y[clean])
        else:
            if history is None:
                history = json.loads(_ARCHIVE.read_text())
            for previous in reversed(history):
                hx, hy = _points(previous, lab)
                keep = _clean(hx, hy)
                if keep.sum() >= 2 and np.ptp(hx[keep]) > 0:
                    out[lab] = _affine(hx[keep], hy[keep])
                    break
            else:
                raise ValueError(f'No calibration sweep with stress variation for lab {lab}')
    return out


def _align(train, test):
    labs, strata, raw, refs = (np.concatenate((train[k], test[k])) for k in
                               ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        take = labs == lab
        ref = refs[take]
        aligned[take] = ((raw[take] - np.nanmedian(np.nanmean(ref, axis=0))) /
                         np.nanmedian(np.nanstd(ref, axis=0)))
    _, cells = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = []
    for c in np.unique(cells):
        values = aligned[cells == c]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(raw.shape[1], np.nan), where=counts > 0))
    return np.where(np.isfinite(aligned), aligned, np.nanmean(means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return (gain[:, None] * split['past_stress_cmd'] + bias[:, None],
            gain * split['stress_hold_cmd'] + bias)


def _design(past, hold, x, dt, times, taus):
    n = len(hold)
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = -np.expm1(-dt[:, None] / taus) * np.exp(-later[:, None] / taus)
    gap = hold[:, None] - past @ memory
    response = -np.expm1(-times[:, None] / taus)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    phi = times / (1.0 + 0.85 * times)
    drift = np.column_stack((np.ones(n), x, slip))
    diff = gap[:, 0] - gap[:, 1]
    curves = np.concatenate((gap[:, None, :] * response,
                             drift[:, None, :] * phi[None, :, None],
                             (diff[:, None] * response[:, 0])[..., None]), axis=2)
    rate = np.column_stack((gap / taus, drift, diff / taus[0]))
    return curves, rate


def _fit(past, hold, x, dt, times, curves, rates):
    target = np.r_[curves.ravel(), rates]

    def design(logtaus):
        a, b = _design(past, hold, x, dt, times, np.exp(logtaus))
        return np.concatenate((a.reshape(-1, 9), b))

    def predict(d, w):
        return np.einsum('ij,j->i', d, w)

    def residual(params):
        return predict(design(params[:2]), params[2:]) - target

    def jacobian(params):
        d = design(params[:2])
        h = 1.e-5
        # Only the two memory coordinates require numerical derivatives.
        columns = [predict((design(params[:2] + step) - design(params[:2] - step)) / (2*h),
                           params[2:]) for step in ([h, 0.], [0., h])]
        return np.column_stack((*columns, d))

    lo, hi = np.log(dt.min() / 30), np.log(times.max() * 30)
    bounds = (np.r_[lo, lo, np.full(9, -np.inf)], np.r_[hi, hi, np.full(9, np.inf)])
    starts = [(0.16, 1.0), (0.4, 2.5), (0.9, 5.0)]
    starts += [(b, a) for a, b in starts]

    def optimize(params, scale, loss):
        return least_squares(lambda q: residual(q) / scale, params,
                             jac=lambda q: jacobian(q) / np.broadcast_to(scale, target.shape)[:, None],
                             bounds=bounds, x_scale='jac', loss=loss, max_nfev=120,
                             ftol=1.e-9)

    fits = []
    for start in starts:
        logtaus = np.clip(np.log(start), lo + 1.e-8, hi - 1.e-8)
        weights = lstsq(design(logtaus), target, lapack_driver='gelsy')[0]
        fits.append(optimize(np.r_[logtaus, weights], .05, 'soft_l1'))
    seed = min(fits, key=lambda r: r.cost).x
    err = residual(seed)
    def mad(values):
        return max(1.e-6, 1.4826 * np.median(np.abs(values - np.median(values))))
    scale = 2.5 * np.r_[np.full(curves.size, mad(err[:curves.size])),
                       np.full(rates.size, mad(err[curves.size:]))]
    best = optimize(seed, scale, 'cauchy').x
    return np.exp(best[:2]), best[2:]


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    cal = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    taus, weights = _fit(*_physical(train, cal), aligned[:n], dt, times,
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    a, b = _design(*_physical(test, cal), aligned[n:], dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
