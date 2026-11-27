"""Calibrated two-branch relaxation with a strain-crystallized plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _sweep(record, channel):
    lab, cmd, true = (np.asarray(record[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
    return cmd[take], true[take]


def _consensus(x, y):
    """Find a tightly affine subset separated from the EMI readings."""
    best = None
    for i, j in zip(*np.triu_indices(len(x), 1)):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        err = np.abs(y - (y[i] + gain * (x - x[i])))
        keep = err < 0.005
        if np.any((~keep) & (err <= 3.0)):
            continue
        score = (keep.sum(), -np.sum(err[keep]**2))
        if best is None or score > best[0]:
            best = score, keep
    return None if best is None else best[1]


def _historical_line(history, channel, x, y):
    candidates = []
    for record in history:
        hx, hy = _sweep(record, channel)
        if np.unique(hx).size < 2:
            continue
        fit = _line(hx, hy)
        if np.max(np.abs(hy - np.polyval(fit, hx))) > 0.005:
            keep = _consensus(hx, hy)
            if keep is None or keep.sum() < 4:
                continue
            hx, hy = hx[keep], hy[keep]
            fit = _line(hx, hy)
        err = np.abs(y - np.polyval(fit, x))
        match = err < 0.005
        candidates.append((fit, hx, hy, match.sum(), np.sum((~match) & (err <= 3.0))))
    # Resolve calibration states by surviving current anchors, then prefer the
    # most recent compatible state. Pool only equivalent archived sweeps.
    chosen = max(range(len(candidates)), key=lambda i: (candidates[i][3], -candidates[i][4], i))
    anchor = candidates[chosen][0]
    group = [c for c in candidates if np.max(np.abs(c[0] - anchor)) < 0.003]
    return _line(np.concatenate([c[1] for c in group]), np.concatenate([c[2] for c in group]))


def _calibrate(cal):
    lines, history = [], None
    for channel in range(4):
        x, y = _sweep(cal, channel)
        if np.unique(x).size >= 2:
            fit = _line(x, y)
            if np.max(np.abs(y - np.polyval(fit, x))) <= 0.005:
                lines.append(fit)
                continue
            keep = _consensus(x, y)
            if keep is not None and keep.sum() >= 4:
                lines.append(_line(x[keep], y[keep]))
                continue
            if keep is None or keep.all():
                lines.append(fit)
                continue
        if history is None:
            history = json.loads(_HISTORY.read_text())
        lines.append(_historical_line(history, channel, x, y))
    return np.array(lines)


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        rows = labs == lab
        reference = panel[rows]
        reference = reference[:, np.isfinite(reference).any(axis=0)]
        center = np.median(np.nanmean(reference, axis=0))
        scale = np.median(np.nanstd(reference, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.empty((len(cells), raw.shape[1]))
    for i in range(len(cells)):
        values = aligned[ids == i]
        count = np.isfinite(values).sum(axis=0)
        means[i] = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(raw.shape[1], np.nan), where=count > 0)
    fills = means.copy()
    for i, j in zip(*np.where(~np.isfinite(means))):
        available = np.isfinite(means[:, j])
        same_lab = available & (cells[:, 0] == cells[i, 0])
        use = same_lab if same_lab.any() else available
        fills[i, j] = means[use, j].mean() if use.any() else 0.0
    completed = np.where(np.isfinite(aligned), aligned, fills[ids])
    n_train = len(train['lab_id'])
    return completed[:n_train], completed[n_train:]


def _inputs(part, aligned, lines, dt, threshold, crystallizes):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    damage = np.maximum(np.abs(stress) - threshold, 0.0)**2 @ dt
    activation = np.tanh(damage) if crystallizes else np.zeros_like(damage)
    coupons = np.column_stack((np.ones(len(hold)), aligned))
    return stress, hold, coupons, activation


def _design(logtau, inputs, dt, times, crystallizes):
    stress, hold, coupons, activation = inputs
    inverse = np.exp(-logtau)
    # Unroll the stated zero-initial-memory recurrence into its stress kernel.
    suffix = np.r_[np.cumsum(dt[::-1])[::-1][1:], 0.0]
    kernel = -np.expm1(-dt[:, None] * inverse) * np.exp(-suffix[:, None] * inverse)
    memory = stress @ kernel
    gap = hold[:, None] - memory
    exp_basis = np.vstack((-np.expm1(-times[:, None] * inverse), inverse))
    # In this phase, damage activates the two corrections, not the base
    # logarithmic creep. The base creep is continuous through zero damage.
    flow = np.r_[np.log1p(times) if crystallizes else times, 1.0]
    a = np.empty((len(hold), len(times)+1, 9))
    a[:, :, :2] = gap[:, None, :] * exp_basis[None, :, :]
    a[:, :, 2:7] = coupons[:, None, :] * flow[None, :, None]
    a[:, :, 7] = (activation * hold)[:, None] * exp_basis[:, 0]
    a[:, :, 8] = (activation * memory[:, 0])[:, None] * flow
    return a


def _fit(inputs, dt, times, target, crystallizes, complete=None):
    valid = np.isfinite(target)
    # Imputed coupons are useful for prediction but introduce errors in the
    # regressors. Fit to complete histories when they identify the design.
    if complete is not None:
        use = valid & complete[:, None]
        probe = _design(np.log([0.5, 3.0]), inputs, dt, times, crystallizes)[use]
        n_weights = 9 if crystallizes and np.any(inputs[-1]) else 7
        if len(probe) >= 3 * n_weights and np.linalg.matrix_rank(probe) >= n_weights:
            valid = use
    y = target[valid]
    sqrt_weight = np.ones_like(y)

    def profile(logtau, coefficients=False):
        a = _design(logtau, inputs, dt, times, crystallizes)[valid] * sqrt_weight[:, None]
        wy = y * sqrt_weight
        w = np.linalg.lstsq(a, wy, rcond=None)[0]
        return w if coefficients else a @ w - wy

    def optimize(start):
        return least_squares(profile, start, bounds=(np.log(0.025), np.log(60.0)),
                             ftol=1e-9, gtol=1e-9, xtol=1e-9, max_nfev=100)

    starts = ((0.18, 1.0), (0.55, 3.5), (1.5, 8.0), (3.5, 0.55))
    opt = min((optimize(np.log(s)) for s in starts), key=lambda r: r.cost).x
    # Use a Cauchy influence function, with a separate measurement scale for
    # each sampled time and the initial-rate measurement.
    for _ in range(8):
        resid = _design(opt, inputs, dt, times, crystallizes) @ profile(opt, True) - target
        resid = np.where(valid, resid, np.nan)
        effective = valid.any(axis=0)
        r = resid[:, effective]
        mad = np.nanmedian(np.abs(r - np.nanmedian(r, axis=0)), axis=0)
        scale = np.full(target.shape[1], 1.0)
        scale[effective] = np.maximum(2.385 * 1.4826 * mad, 1e-6)
        sqrt_weight = (1.0 / np.hypot(1.0, resid / scale))[valid]
        opt = optimize(opt).x
    if not crystallizes or not np.any(inputs[-1]):
        opt.sort()
    return opt, profile(opt, True)


def solve_alpha_m97_c03_batch_07(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    lines = _calibrate(dataset['calibration_load_cells'])
    aligned_train, aligned_test = _align(train, test)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    crystallizes = dataset['shear_banding_index'] <= 0.50
    args = dt, dataset['crit_stress'], crystallizes
    inputs = _inputs(train, aligned_train, lines, *args)
    target = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                              train['instant_rate_obs']))
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    logtau, weights = _fit(inputs, dt, times, target, crystallizes, complete)
    ans = _design(logtau, _inputs(test, aligned_test, lines, *args), dt, times, crystallizes) @ weights
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': ans[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': ans[:, -1]}
