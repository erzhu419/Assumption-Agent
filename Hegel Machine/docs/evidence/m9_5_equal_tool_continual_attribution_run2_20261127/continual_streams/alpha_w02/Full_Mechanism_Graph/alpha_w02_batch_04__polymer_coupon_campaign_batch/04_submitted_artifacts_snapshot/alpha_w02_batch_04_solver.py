"""Load-cell calibrated, preloaded hereditary creep for polymer coupons."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _checks(checks, lab):
    labs, x, y = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    m = (labs == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _consensus(x, y):
    # An EMI spike cannot define the line when three or more checks agree.
    i, j = np.triu_indices(len(x), 1)
    good = np.abs(x[i]-x[j]) > 1e-8
    i, j = i[good], j[good]
    if not len(i):
        return []
    g = (y[j]-y[i])/(x[j]-x[i])
    r = (y[None, :]-y[i, None])-g[:, None]*(x[None, :]-x[i, None])
    keep = np.abs(r) <= .001
    masks = np.unique(keep[keep.sum(axis=1) == keep.sum(axis=1).max()], axis=0)
    fits = []
    for m in masks:
        fit = np.linalg.lstsq(np.c_[x[m], np.ones(m.sum())], y[m], rcond=None)[0]
        fits.append((fit, m.sum(), np.ptp(x[m])))
    return fits


def _archive_checks(root):
    path = root / 'archive_calibration_history.json'
    if path.is_file():
        yield from reversed(json.loads(path.read_text()))


def _with_prior(x, y, fits, prior):
    gain, bias = prior
    if not len(x):
        return prior
    # Fix the slope when a narrow or contaminated spot-check panel cannot
    # estimate it. A shared offset across checks identifies bias drift.
    offsets = y - gain*x - bias
    center = np.median(offsets)
    agreeing = np.abs(offsets-center) <= .004
    if agreeing.sum() > len(x)//2 and (len(x) > 1 or abs(center) < .10):
        bias += np.median(offsets[agreeing])
    else:
        i = np.argmin(np.abs(offsets))
        if abs(offsets[i]) < .10:
            bias += offsets[i]
    for fit, count, span in fits:
        if (count >= 3 and span >= .4 and abs(fit[0]-gain) <= .01
                and abs(fit[1]-bias) <= .01):
            return fit
    return gain, bias


def _calibrate(checks, root=Path('/tmp/hegel_continual_archive')):
    result = np.tile([1., 0.], (4, 1))
    unresolved = {}
    for lab in range(4):
        x, y = _checks(checks, lab)
        fits = _consensus(x, y)
        if len(x) >= 4 and len(fits) == 1 and fits[0][1] >= 3 and fits[0][2] >= .1:
            result[lab] = fits[0][0]
        else:
            unresolved[lab] = x, y, fits
    if unresolved:
        for table in _archive_checks(root):
            for lab in list(unresolved):
                xa, ya = _checks(table, lab)
                prior = _consensus(xa, ya)
                if len(prior) != 1 or prior[0][1] < 3 or prior[0][2] < .1:
                    continue
                x, y, fits = unresolved.pop(lab)
                result[lab] = _with_prior(x, y, fits, prior[0][0])
            if not unresolved:
                break
    for lab, (x, y, fits) in unresolved.items():
        supported = [f for f in fits if f[2] >= .1]
        if supported:
            result[lab] = min(supported, key=lambda f: np.linalg.norm((f[0]-[1., 0.])*[2., 1.]))[0]
        elif len(x):
            result[lab, 1] = np.median(y-x)
    return result


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    reference = np.vstack([s['ref_coupons'] for s in (train, test)])
    x = np.vstack([s['coupon_features_raw'] for s in (train, test)]).astype(float)
    for c in np.unique(lab):
        rows = lab == c
        r = reference[rows]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        x[rows] = (x[rows] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    _, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    cell = cell.ravel()
    means = np.array([np.nanmean(x[cell == k], axis=0) for k in np.unique(cell)])
    fill = np.nanmean(means, axis=0)
    return np.column_stack((np.ones(len(x)), np.where(np.isfinite(x), x, fill)))


def _stress(split, calib):
    gain, bias = np.asarray(calib)[split['lab_id']].T
    return (gain[:, None] * split['past_stress_cmd'] + bias[:, None],
            gain * split['stress_hold_cmd'] + bias)


def _plastic(stress, dt, yield_stress):
    past, hold = stress
    excess = np.maximum(np.abs(hold) - yield_stress, 0.)
    work = np.maximum(np.abs(past) - yield_stress, 0.)**2 @ dt
    return excess**2 * np.sign(hold), work * hold


def _design(stress, feat, plastic, dt, time, tau):
    past, hold = stress
    end = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    decay = np.exp(-end[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    amp = hold[:, None] - past @ decay
    rise = -np.expm1(-time[:, None] / tau)
    curve = [amp[:, None, :] * rise[None, :, :], feat[:, None, :] * time[None, :, None]]
    rate = [amp / tau, feat]
    if plastic is not None:
        over, hist = plastic
        curve.extend([(over[:, None] * time)[..., None], (hist[:, None] * rise[:, 0])[..., None]])
        rate.extend([over[:, None], (hist / tau[0])[:, None]])
    return np.concatenate((np.concatenate(curve, axis=2), np.column_stack(rate)[:, None, :]), axis=1)


def _fit(stress, feat, plastic, dt, time, y):
    mask = np.isfinite(y)
    weight = np.ones(y.shape)
    q = 7 if plastic is None else 9

    def fit_at(log_tau, full=False):
        tau = np.exp(np.sort(log_tau))
        design = _design(stress, feat, plastic, dt, time, tau)
        a = design[mask] * weight[mask, None]
        target = y[mask] * weight[mask]
        coef = np.linalg.lstsq(a, target, rcond=1e-7)[0]
        if full:
            return tau, coef, design @ coef
        return a @ coef - target

    def optimize(start):
        return least_squares(fit_at, np.log(start), bounds=(np.log(.05001), np.log(24.999)),
                             max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)

    result = min((optimize(start) for start in ((.35, 2.), (.8, 4.), (1.3, 8.))), key=lambda f: f.cost)
    tau, coef, pred = fit_at(result.x, True)
    # Profile linear weights inside each nonlinear solve; downweight individual
    # spikes without discarding the rest of the coupon's measurements.
    for iteration in range(12):
        residual = y - pred
        scales = 1.4826 * np.array([np.nanmedian(np.abs(residual[:, :-1])),
                                   np.nanmedian(np.abs(residual[:, -1]))])
        scales = np.maximum(scales, 1e-8)
        scale = np.r_[np.repeat(scales[0], y.shape[1] - 1), np.clip(scales[1], scales[0] / 10, scales[0] * 10)]
        cutoff = 2.5 * scale
        weight = (scales[0] / scale) / np.hypot(1., np.where(mask, residual, 0.) / cutoff)
        result = optimize(tau)
        next_tau, next_coef, next_pred = fit_at(result.x, True)
        change = np.max(np.abs(next_pred - pred))
        tau, coef, pred = next_tau, next_coef, next_pred
        if iteration >= 2 and change < 1e-7 * max(1., np.max(np.abs(pred))):
            break
    return tau, coef.reshape(q)


def solve_alpha_w02_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    calib = _calibrate(dataset['calibration_load_cells'])
    train_stress, test_stress = (_stress(s, calib) for s in (train, test))
    yield_stress = dataset['yield_stress']
    train_plastic, test_plastic = (_plastic(s, dt, yield_stress) for s in (train_stress, test_stress))
    if not any(np.any(x) for x in train_plastic):
        train_plastic = test_plastic = None
    feat = _features(train, test)
    n = len(train['lab_id'])
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']))
    tau, coef = _fit(train_stress, feat[:n], train_plastic, dt, time, y)
    pred = _design(test_stress, feat[n:], test_plastic, dt, time, tau) @ coef
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'viscoplastic_weights': coef[7:] if len(coef) > 7 else np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
