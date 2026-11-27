"""Calibrated two-branch viscoelastic model with jammed slip."""
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


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _design(logtau, inputs, dt, times):
    stress, hold, coupons = inputs
    tau = np.exp(logtau)
    memory = np.zeros((len(hold), 2))
    for step, duration in enumerate(dt):
        increment = -np.expm1(-duration / tau)
        memory += (stress[:, step, None] - memory) * increment
    gap = hold[:, None] - memory
    response = np.vstack((-np.expm1(-times[:, None] / tau), 1 / tau))
    flow = np.r_[times / (1 + 0.85 * times), 1.0]
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    a = np.empty((len(hold), len(flow), 9))
    a[:, :, :2] = gap[:, None, :] * response[None, :, :]
    a[:, :, 2:7] = coupons[:, None, :] * flow[None, :, None]
    a[:, :, 7] = slip[:, None] * flow
    a[:, :, 8] = (gap[:, 0] - gap[:, 1])[:, None] * response[:, 0]
    return a


def _fit(inputs, dt, times, y):
    valid = np.isfinite(y)
    target = y[valid]
    sqrt_weight = np.ones_like(target)

    def profile(logtau, coefficients=False):
        a = _design(logtau, inputs, dt, times)[valid] * sqrt_weight[:, None]
        b = target * sqrt_weight
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        return w if coefficients else a @ w - b

    bounds = np.log((0.06, 30.0))

    def optimize(start):
        return least_squares(profile, start, bounds=tuple(bounds), max_nfev=90,
                             ftol=1e-9, gtol=1e-9, xtol=1e-9)

    starts = ((0.45, 2.4), (1.2, 6.0), (2.4, 0.45), (0.22, 1.3))
    logtau = min((optimize(np.log(s)) for s in starts), key=lambda fit: fit.cost).x
    for _ in range(8):
        resid = _design(logtau, inputs, dt, times) @ profile(logtau, True) - y
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0)
        scale = np.maximum(2.385 * 1.4826 * mad, 1e-6)
        sqrt_weight = (1 / np.hypot(1, resid / scale))[valid]
        logtau = optimize(logtau).x
    return logtau, profile(logtau, True)


def solve_alpha_m97_c03_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    lines = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    logtau, w = _fit(_inputs(train, x_train, lines), dt, times, y)
    pred = _design(logtau, _inputs(test, x_test, lines), dt, times) @ w
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
