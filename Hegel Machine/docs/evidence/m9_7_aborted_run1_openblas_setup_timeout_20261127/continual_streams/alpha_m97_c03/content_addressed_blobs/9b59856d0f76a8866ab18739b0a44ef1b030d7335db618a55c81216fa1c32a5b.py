"""Calibrate and fit the two-branch jammed-slip rheology model."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sweep(cal, lab):
    x = np.asarray(cal['cmd_stress'], dtype=float)
    y = np.asarray(cal['true_stress'], dtype=float)
    rows = (np.asarray(cal['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[rows], y[rows]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _calibrate(cal):
    params = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _sweep(cal, lab)
        spread = np.unique(x).size >= 2
        if x.size >= 2 and not spread:
            raise ValueError(f'Load cell {lab} has no calibration stress spread')
        spike = False
        if x.size >= 3:
            # A three-point fit can absorb a spike. Test each observation
            # against the line fitted to the remaining sweep points.
            for i in range(x.size):
                keep = np.arange(x.size) != i
                if np.unique(x[keep]).size >= 2:
                    gain, bias = _line(x[keep], y[keep])
                    if abs(y[i] - (gain*x[i] + bias)) > 3.0:
                        spike = True
                        break
        if spread and not spike:
            params[lab] = _line(x, y)
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text())
        for saved in reversed(history):
            hx, hy = _sweep(saved, lab)
            if np.unique(hx).size >= 2:
                prior = _line(hx, hy)
                break
        else:
            raise ValueError(f'No archived calibration for load cell {lab}')
        clean = np.abs(y - (prior[0]*x + prior[1])) <= 3.0
        if clean.sum() >= 4 and np.unique(x[clean]).size >= 2:
            params[lab] = _line(x[clean], y[clean])
        elif not spike or not clean.all():
            params[lab] = prior
        else:
            # The archive does not corroborate EMI; retain in-batch data.
            params[lab] = _line(x, y)
    return params


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        mask = labs == lab
        refs = panel[mask]
        refs = refs[:, np.any(np.isfinite(refs), axis=0)]
        center = np.median(np.nanmean(refs, axis=0))
        scale = np.median(np.nanstd(refs, axis=0, ddof=0))
        aligned[mask] = (raw[mask] - center) / scale
    cells = np.unique(np.column_stack((labs, strata)), axis=0)
    masks = [(labs == lab) & (strata == s) for lab, s in cells]
    means = np.empty((len(cells), aligned.shape[1]))
    for i, mask in enumerate(masks):
        rows = aligned[mask]
        counts = np.isfinite(rows).sum(axis=0)
        means[i] = np.divide(np.nansum(rows, axis=0), counts,
                             out=np.full(rows.shape[1], np.nan), where=counts > 0)
    for i, mask in enumerate(masks):
        fill = means[i].copy()
        for j in np.flatnonzero(~np.isfinite(fill)):
            local = means[cells[:, 0] == cells[i, 0], j]
            global_ = means[:, j]
            available = local[np.isfinite(local)]
            if not available.size:
                available = global_[np.isfinite(global_)]
            fill[j] = available.mean() if available.size else 0.0
        aligned[mask] = np.where(np.isfinite(aligned[mask]), aligned[mask], fill)
    return aligned


def _physical(split, params):
    gain, bias = params[split['lab_id']].T
    return (split['past_stress_cmd'] * gain[:, None] + bias[:, None],
            split['stress_hold_cmd'] * gain + bias)


def _design(tau, past, hold, dt, times, coupon):
    # Unroll the recurrence: each increment survives all later intervals.
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    step = dt[:, None] / tau
    kernel = -np.expm1(-step) * np.exp(-after[:, None] / tau)
    gap = hold[:, None] - past @ kernel
    phi = np.r_[times / (1.0 + 0.85*times), 1.0]
    prony = np.vstack((-np.expm1(-times[:, None] / tau), 1.0 / tau))
    design = np.empty((hold.size, phi.size, 9))
    design[:, :, :2] = gap[:, None, :] * prony
    design[:, :, 2] = phi
    design[:, :, 3:7] = coupon[:, None, :] * phi[None, :, None]
    design[:, :, 7] = (np.sign(hold)*np.sqrt(np.abs(hold)))[:, None] * phi
    design[:, :, 8] = (gap[:, 0] - gap[:, 1])[:, None] * prony[:, 0]
    return design


def _fit(past, hold, dt, times, coupon, y):
    args = past, hold, dt, times, coupon
    lo = np.r_[np.log([0.015, 0.04]), np.full(9, -np.inf)]
    hi = np.r_[np.log([20.0, 200.0]), np.full(9, np.inf)]

    def taus(p):
        fast, ratio = np.exp(p[:2])
        return np.array([fast, fast * (1.0 + ratio)])

    def residual(p):
        return ((_design(taus(p), *args) @ p[2:] - y) / 0.015).ravel()

    best = None
    for fast, slow in ((0.25, 1.6), (0.55, 3.2), (1.2, 7.0)):
        design = _design(np.array([fast, slow]), *args).reshape(-1, 9)
        weights = np.linalg.lstsq(design, y.ravel(), rcond=None)[0]
        start = np.r_[np.log([fast, slow / fast - 1.0]), weights]
        # Soft L1 gets close before Cauchy downweights large isolated spikes.
        for loss, width in (('soft_l1', 3.0), ('cauchy', 1.0)):
            fit = least_squares(residual, start, loss=loss, f_scale=width,
                                bounds=(lo, hi), max_nfev=180, x_scale='jac')
            start = fit.x
        if best is None or fit.cost < best.cost:
            best = fit
    return taus(best.x), best.x[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    params = _calibrate(dataset['calibration_load_cells'])
    coupon = _align(train, test)
    dt, times = dataset['past_dt'], dataset['eval_times']
    n = len(train['lab_id'])
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    tau, w = _fit(*_physical(train, params), dt, times, coupon[:n], y)
    pred = _design(tau, *_physical(test, params), dt, times, coupon[n:]) @ w
    return dict(estimated_taus=tau, hereditary_weights=w[:7], regime_mode_weights=w[7:],
                predicted_strain=test['strain_0'][:, None] + pred[:, :-1],
                initial_rate_pred=pred[:, -1])
