"""Calibrated two-branch rheology with jammed slip."""
import json
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate([train[k], test[k]]) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty_like(raw)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _calibrate(cal):
    result = np.empty((4, 2))
    archive = None
    for lab in range(4):
        rows = np.asarray(cal['lab_id']) == lab
        x, y = (np.asarray(cal[k])[rows] for k in ('cmd_stress', 'true_stress'))
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        a = np.column_stack((x, np.ones(len(x))))
        line = np.linalg.lstsq(a, y, rcond=None)[0]
        # Clean sweeps are linear to the measurement noise. If inconsistent,
        # look for a clean consensus with a large off-line reading.
        emi = np.any(np.abs(y - x) > 3.0)
        if len(x) >= 3 and np.max(np.abs(y - a @ line)) > 0.001 and not emi:
            for i, j in combinations(range(len(x)), 2):
                if abs(x[i] - x[j]) < 0.1:
                    continue
                slope = (y[j] - y[i]) / (x[j] - x[i])
                if not 0.5 < slope < 1.6:
                    continue
                err = np.abs(y - (slope * (x - x[i]) + y[i]))
                if (err < 0.001).sum() >= max(2, len(x) // 2 + 1) and np.any(err > 3.0):
                    emi = True
                    break
        if len(x) < 2 or emi:
            if archive is None:
                with open(_ARCHIVE) as f:
                    archive = json.load(f)
            for sweep in reversed(archive):
                mask = np.asarray(sweep['lab_id']) == lab
                if mask.sum() >= 2:
                    sx, sy = (np.asarray(sweep[k])[mask] for k in ('cmd_stress', 'true_stress'))
                    prior = np.polyfit(sx, sy, 1)
                    break
            keep = np.abs(y - a @ prior) <= 3.0
            if keep.sum() < 2 or np.ptp(x[keep]) < 0.1:
                line = prior
            elif emi:
                line = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
        result[lab] = line
    return result


def _design(sample, aligned, calibration, dt, times):
    gain, bias = calibration[np.asarray(sample['lab_id'])].T
    history = gain[:, None] * sample['past_stress_cmd'] + bias[:, None]
    hold = gain * sample['stress_hold_cmd'] + bias
    flow = np.column_stack((np.ones(len(hold)), aligned, np.sign(hold) * np.sqrt(np.abs(hold))))
    tail = times / (1.0 + 0.85 * times)
    flow_curves = flow[:, None, :] * tail[None, :, None]
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]

    def evaluate(taus):
        # Unroll the recurrence, including the decay after each past segment.
        kernel = -np.expm1(-dt[:, None] / taus) * np.exp(-remaining[:, None] / taus)
        gap = hold[:, None] - history @ kernel
        relax = -np.expm1(-times[:, None] / taus)
        cross = gap[:, 0] - gap[:, 1]
        curves = np.concatenate((gap[:, None, :] * relax, flow_curves,
                                 (cross[:, None] * relax[:, 0])[:, :, None]), axis=2)
        rates = np.column_stack((gap / taus, flow, cross / taus[0]))
        return curves, rates

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, times):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    base = np.r_[np.full(cm.sum(), 1 / np.sqrt(len(times))), np.ones(rm.sum())]

    def project(p, w):
        curves, rates = evaluate(np.exp(np.sort(p)))
        matrix = np.concatenate((curves[cm], rates[rm]))
        a = matrix * w[:, None]
        b = y * w
        weights = np.linalg.lstsq(a, b, rcond=None)[0]
        return np.einsum('ij,j->i', a, weights) - b, weights

    bounds = np.log([min(dt.min(), times[times > 0].min()) / 12,
                     max(dt.sum(), times.max()) * 12])
    short, long = np.median(dt), np.median(times)
    starts = [(0.4*short, long), (short, long), (2*short, 2*long), (short, 4*long)]
    def fit(start, w):
        return least_squares(lambda p: project(p, w)[0], start,
                             bounds=bounds, max_nfev=70)
    p = min((fit(np.log(pair), base) for pair in starts), key=lambda f: f.cost).x
    groups = np.r_[np.nonzero(cm)[1], np.full(rm.sum(), len(times))]
    for _ in range(4):
        r = project(p, np.ones_like(base))[0] if _ == 0 else project(p, w)[0] / w
        scale = np.empty_like(y)
        for j in np.unique(groups):
            ix = groups == j
            q = r[ix]
            scale[ix] = max(1e-5, 1.4826 * np.median(np.abs(q - np.median(q))))
        w = base / (scale * np.sqrt(1 + (r / (2.5 * scale))**2))**0.5
        p = fit(p, w).x
    return np.exp(np.sort(p)), project(p, w)[1]


def solve_alpha_m97_c03_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    a_train, a_test = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    fit_rows = {k: v[complete] for k, v in train.items()}
    taus, weights = _fit(_design(fit_rows, a_train[complete], calibration, dt, times),
                         fit_rows['strain_curves_obs'] - fit_rows['strain_0'][:, None],
                         fit_rows['instant_rate_obs'], dt, times)
    curves, rates = _design(test, a_test, calibration, dt, times)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
