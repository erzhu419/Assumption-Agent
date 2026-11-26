"""Fit the two-mode hereditary response of a calibrated coupon batch."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _lab_checks(panel, lab):
    x = np.asarray(panel['cmd_stress'], dtype=float)
    y = np.asarray(panel['true_stress'], dtype=float)
    keep = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _consensus(x, y, prior):
    """Fit a positive-slope affine map to the best-supported local checks."""
    if len(x) < 3 or np.ptp(x) < .1:
        return None
    i, j = np.triu_indices(len(x), 1)
    ok = np.abs(x[j] - x[i]) >= .1
    i, j = i[ok], j[ok]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    offsets = y[i] - slopes * x[i]
    error = np.abs(y - slopes[:, None] * x - offsets[:, None])
    inliers = error <= .001
    support = inliers.sum(axis=1)
    candidates = np.flatnonzero((support >= max(3, len(x)//2 + 1)) & (slopes > 0))
    if not candidates.size:
        return None
    candidates = candidates[support[candidates] == support[candidates].max()]
    loss = np.minimum(error[candidates], .001).sum(axis=1)
    delta = (slopes[candidates] - prior[0])**2 + (offsets[candidates] - prior[1])**2
    best = candidates[np.lexsort((delta, loss))[0]]
    keep = inliers[best]
    return np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]


def _calibration(panel):
    nominal = np.array([1., 0.])
    checks = [_lab_checks(panel, lab) for lab in range(4)]
    maps = [_consensus(x, y, nominal) for x, y in checks]
    if any(m is None for m in maps):
        history = json.loads(_HISTORY.read_text()) if _HISTORY.exists() else []
        for lab, (x, y) in enumerate(checks):
            if maps[lab] is not None:
                continue
            prior = nominal
            for archived in reversed(history):
                m = _consensus(*_lab_checks(archived, lab), nominal)
                if m is not None:
                    prior = m
                    break
            delta = y - prior[0] * x - prior[1]
            close = delta[np.abs(delta) <= .02]
            maps[lab] = prior + [0., np.median(close) if close.size else 0.]
    return np.asarray(maps)


def _alignment(train, test):
    def joined(key):
        return np.concatenate((train[key], test[key]))
    labs, strata = joined('lab_id'), joined('stratum')
    x, refs = joined('coupon_features_raw'), joined('ref_coupons')
    for lab in np.unique(labs):
        rows = labs == lab
        panel = refs[rows]
        cols = np.isfinite(panel).sum(axis=0) >= 2
        if cols.any():
            center = np.median(np.nanmean(panel[:, cols], axis=0))
            scale = np.median(np.nanstd(panel[:, cols], axis=0, ddof=0))
            x[rows] = (x[rows] - center) / (scale if scale > 0 else 1.)
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    finite = np.isfinite(x)
    count = np.zeros((cell.max() + 1, x.shape[1]))
    total = np.zeros_like(count)
    np.add.at(count, cell, finite)
    np.add.at(total, cell, np.where(finite, x, 0.))
    means = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    fill = means.sum(axis=0) / np.maximum(1, (count > 0).sum(axis=0))
    n = len(train['lab_id'])
    return np.where(finite, x, fill)[:n], np.where(finite, x, fill)[n:]


def _inputs(block, x, maps):
    gain, bias = maps[np.asarray(block['lab_id'])].T
    hist = block['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = block['stress_hold_cmd'] * gain + bias
    return hist, hold, np.asarray(block['initial_memory_states']), np.column_stack((np.ones(len(x)), x))


def _design(taus, inputs, dt, time):
    hist, hold, initial, cov = inputs
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = np.exp(-tail[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    onset = hist @ memory + initial * np.exp(-dt.sum() / taus)
    jump = hold[:, None] - onset
    relax = jump[:, None, :] * -np.expm1(-time[:, None] / taus)
    drift = cov[:, None, :] * time[None, :, None]
    return np.concatenate((relax, drift), axis=2), np.column_stack((jump / taus, cov))


def _mad(e):
    return max(1e-7, 1.4826 * np.median(np.abs(e - np.median(e))))


def _fit(train, inputs, dt, time):
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    if complete.sum() < 12:
        complete[:] = True
    inputs = tuple(a[complete] for a in inputs)
    inc = (train['strain_curves_obs'] - train['strain_0'][:, None])[complete]
    y = np.r_[inc.ravel(), train['instant_rate_obs'][complete]]
    width = inputs[-1].shape[1] + 2

    def design(taus):
        a, b = _design(taus, inputs, dt, time)
        return np.vstack((a.reshape(-1, width), b))

    def unpack(p):
        return np.array([p[0], p[0] + (24.99 - p[0]) * p[1]])

    def residual(p):
        return np.einsum('ij,j->i', design(unpack(p)), p[2:]) - y

    bounds = (np.r_[.051, 1e-5, np.full(width, -np.inf)],
              np.r_[24.98, 1., np.full(width, np.inf)])
    fits = []
    for fast, slow in ((.35, 2.), (.65, 4.5), (1.1, 8.), (2., 14.)):
        A = design([fast, slow])
        start = np.r_[fast, (slow-fast)/(24.99-fast), np.linalg.lstsq(A, y, rcond=None)[0]]
        fits.append(least_squares(residual, start, bounds=bounds, loss='soft_l1', f_scale=.04, max_nfev=100))
    p = min(fits, key=lambda fit: fit.cost).x
    for _ in range(2):
        e = residual(p)
        scale = np.r_[np.full(inc.size, 2.5 * _mad(e[:inc.size])),
                      np.full(inc.shape[0], 2.5 * _mad(e[inc.size:]))]
        p = least_squares(lambda q: residual(q)/scale, p, bounds=bounds, loss='cauchy', max_nfev=100).x
    return unpack(p), p[2:]


def _crossing(time, initial, strain, critical):
    grid = np.r_[0., time]
    distance = np.column_stack((initial, strain)) - critical[:, None]
    reached = distance >= 0
    first = np.argmax(reached, axis=1)
    crossing = np.full(len(initial), time[-1])
    crossing[reached[:, 0]] = 0.
    i = np.flatnonzero(first > 0)
    j = first[i]
    before, after = distance[i, j-1], distance[i, j]
    crossing[i] = grid[j-1] - before / (after-before) * (grid[j]-grid[j-1])
    return crossing


def solve_alpha_w01_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    maps = _calibration(dataset['calibration_load_cells'])
    x, xt = _alignment(train, test)
    taus, w = _fit(train, _inputs(train, x, maps), dt, time)
    inputs = _inputs(test, xt, maps)
    design, onset = _design(taus, inputs, dt, time)
    initial = np.asarray(test['strain_0'])
    strain = initial[:, None] + design @ w
    # Curve residuals estimate small lab offsets; onset rates also constrain drift.
    train_design, _ = _design(taus, _inputs(train, x, maps), dt, time)
    train_error = train['strain_curves_obs'] - train['strain_0'][:, None] - train_design @ w
    late = np.median(train_error[:, -6:] / time[-6:], axis=1)
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    offset = np.zeros(len(test['lab_id']))
    for lab in np.unique(train['lab_id'][complete]):
        offset[test['lab_id'] == lab] = np.median(late[complete & (train['lab_id'] == lab)])
    rate = np.asarray(test['instant_rate_obs'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    observed = np.isfinite(rate)
    drift = offset.copy()
    drift[observed] += .25 * ((rate - onset @ w)[observed] - offset[observed])
    drift[missing & observed] = (rate - onset @ w)[missing & observed]
    strain += drift[:, None] * time
    return {'estimated_taus': taus, 'predicted_strain': strain,
            'secant_compliance': (strain[:, -1] - initial) / np.maximum(np.abs(inputs[1]), .5),
            'hazard_crossing_times': _crossing(time, initial, strain, np.asarray(test['critical_strain']))}
