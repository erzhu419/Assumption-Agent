"""Fit the damage-gated Prony/log-creep model to the current rheology batch."""
import json
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate((train[k], test[k])) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[mask] = (raw[mask] - center) / scale
    cells, member = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.full((len(cells), aligned.shape[1]), np.nan)
    for i in range(len(cells)):
        block = aligned[member == i]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, (lab, _) in enumerate(cells):
        rows = member == i
        fill = means[i].copy()
        for j in np.flatnonzero(~np.isfinite(fill)):
            pool = means[cells[:, 0] == lab, j]
            if not np.isfinite(pool).any():
                pool = means[:, j]
            fill[j] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        aligned[rows] = np.where(np.isfinite(aligned[rows]), aligned[rows], fill)
    return np.split(aligned, [len(train['lab_id'])])


def _sweep(rows, lab):
    x, y = (np.asarray(rows[k], dtype=float)[np.asarray(rows['lab_id']) == lab]
            for k in ('cmd_stress', 'true_stress'))
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _consensus(x, y):
    """Identify the clean affine subset of a precision calibration sweep."""
    best = np.zeros(len(x), dtype=bool)
    best_key = (-1, -np.inf)
    for i, j in combinations(range(len(x)), 2):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        if gain <= 0:
            continue
        bias = y[i] - gain * x[i]
        keep = np.abs(y - (gain * x + bias)) < 0.001
        key = (int(keep.sum()), -abs(np.log(gain)) - abs(bias))
        if key > best_key:
            best, best_key = keep, key
    return best


def _calibrate(rows):
    coeff = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _sweep(rows, lab)
        keep = _consensus(x, y)
        if keep.sum() >= 4:
            coeff[lab] = np.polyfit(x[keep], y[keep], 1)
            continue
        # A nominal-map deviation also detects sparse EMI sweeps in which
        # two readings alone cannot distinguish a spike from an affine line.
        spike = np.any(np.abs(y - x) > 3.0)
        if keep.sum() >= 2:
            line = np.polyfit(x[keep], y[keep], 1)
            spike |= np.any(np.abs(y - np.polyval(line, x)) > 3.0)
        if len(x) >= 2 and not spike:
            coeff[lab] = np.polyfit(x, y, 1)
            continue
        if history is None:
            with open(_ARCHIVE) as stream:
                history = json.load(stream)
        for sweep in reversed(history):
            hx, hy = _sweep(sweep, lab)
            clean = _consensus(hx, hy)
            if clean.sum() >= 4:
                prior = np.polyfit(hx[clean], hy[clean], 1)
                break
        else:
            raise ValueError(f'No clean calibration sweep for channel {lab}')
        residual = y - np.polyval(prior, x)
        usable = np.abs(residual) <= 3.0
        if usable.sum() >= 2 and np.ptp(x[usable]) > 0:
            coeff[lab] = np.polyfit(x[usable], y[usable], 1)
        else:
            coeff[lab] = prior
    return coeff


def _design(sample, aligned, calibration, dt, time, crit, band):
    gain, bias = calibration[sample['lab_id']].T
    past = gain[:, None] * sample['past_stress_cmd'] + bias[:, None]
    hold = gain * sample['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(past) - crit, 0.0)**2 @ dt
    active = (band <= 0.50) & (damage > 0)
    act = np.where(active, np.tanh(damage), 0.0)
    drift = np.column_stack((np.ones(len(hold)), aligned))
    # The low-banding regime sets the creep basis; damage gates only its corrections.
    terminal = np.log1p(time) if band <= 0.50 else time
    after = np.cumsum(dt[::-1])[::-1] - dt
    curves_static = terminal[None, :, None] * drift[:, None, :]

    def evaluate(tau):
        # The closed-form convolution is the stated zero-initial-state recurrence.
        kernel = np.exp(-after[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
        z = past @ kernel
        gap = hold[:, None] - z
        q = -np.expm1(-time[:, None] / tau)
        surge = act * hold
        plateau = act * z[:, 0]
        curves = np.concatenate((gap[:, None, :] * q[None, :, :], curves_static,
                                 (surge[:, None] * q[None, :, 0])[:, :, None],
                                 (plateau[:, None] * terminal)[:, :, None]), axis=2)
        rates = np.column_stack((gap / tau, drift, surge / tau[0], plateau))
        return curves, rates

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, time):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    groups = np.r_[np.nonzero(cm)[1], np.full(rm.sum(), len(time))]
    ratio = np.r_[np.ones(cm.sum()), np.full(rm.sum(), np.sqrt(len(time)))]
    tau_min = max(1e-4, min(dt.min(), time[time > 0].min()) / 20)
    tau_max = 20 * max(dt.sum(), time.max())
    bounds = (np.log(tau_min), np.log(tau_max))

    def project(logt, w):
        curves, rates = evaluate(np.exp(np.sort(logt)))
        m = np.concatenate((curves[cm], rates[rm]), axis=0)
        coef = np.linalg.lstsq(m * w[:, None], y * w, rcond=None)[0]
        return m @ coef - y, coef

    def fit(start, w):
        return least_squares(lambda p: project(p, w)[0] * w, start,
                             bounds=bounds, max_nfev=80)

    step = max(np.median(dt), time.max() / 10)
    starts = [(0.5 * step, 4 * step), (step, 8 * step), (0.3 * step, time.max())]
    sol = min((fit(np.log(pair), ratio) for pair in starts), key=lambda s: s.cost)
    p, w = sol.x, ratio
    indices = [groups == i for i in np.unique(groups)]
    for _ in range(7):
        residual = project(p, w)[0]
        scale = np.empty_like(y)
        for ix in indices:
            r = residual[ix]
            scale[ix] = max(1e-5, 1.4826 * np.median(np.abs(r - np.median(r))))
        w = ratio / (scale * np.sqrt(1 + (residual / (2.5 * scale))**2))**0.5
        p = fit(p, w).x
    return np.exp(np.sort(p)), project(p, w)[1]


def solve_alpha_m97_c03_batch_07(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    dt, time = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    crit, band = (dataset[k] for k in ('crit_stress', 'shear_banding_index'))
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    measured = {k: train[k][complete] for k in ('lab_id', 'past_stress_cmd', 'stress_hold_cmd')}
    taus, weights = _fit(_design(measured, x_train[complete], calibration, dt, time, crit, band),
                         train['strain_curves_obs'][complete] - train['strain_0'][complete, None],
                         train['instant_rate_obs'][complete], dt, time)
    curves, rates = _design(test, x_test, calibration, dt, time, crit, band)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
