"""Calibrate the paired four-channel load-cell bridge."""
from __future__ import annotations

import numpy as np

_PAIR = np.array([2, 3, 0, 1])
# Shared frequencies identified from the bridge rows and single-axis sweeps.
_OWN_FREQ = 2.5
_PAIR_FREQ = 2.0
_HISTORY_FIELDS = ('past_channel_raw', 'hold_channel_raw')


def _design(u, v):
    return np.stack((np.ones_like(u), u, np.sin(_OWN_FREQ * u),
                     v, 1 - np.cos(_PAIR_FREQ * v)), axis=-1)


def _fit(calibration):
    x = np.asarray(calibration['true_channel_loads'], dtype=float)
    y = np.asarray(calibration['bridge_raw'], dtype=float)
    sweep = calibration['single_channel_sweeps']
    lab = np.asarray(sweep['lab_id'])
    sx = np.asarray(sweep['true_stress'], dtype=float)
    sy = np.asarray(sweep['cmd_stress'], dtype=float)
    bridge_design = _design(x, x[:, _PAIR])
    sweep_design = _design(sx, np.zeros_like(sx))
    row_weight, sweep_weight = np.ones(len(x)), np.ones(len(sx))
    coef = np.empty((4, 5))
    for _ in range(8):
        for j in range(4):
            s = lab == j
            a = np.concatenate((bridge_design[:, j], sweep_design[s]))
            b = np.concatenate((y[:, j], sy[s]))
            w = np.sqrt(np.concatenate((row_weight, sweep_weight[s])))
            coef[j] = np.linalg.lstsq(a * w[:, None], b * w, rcond=None)[0]
        # A gross spike compromises the calibration row, not four independent fits.
        row_residual = np.linalg.norm(np.einsum('njk,jk->nj', bridge_design, coef) - y, axis=1)
        sweep_residual = np.abs(np.einsum('nk,nk->n', sweep_design, coef[lab]) - sy)
        row_weight = _robust_weight(row_residual, row_residual)
        sweep_weight = _robust_weight(sweep_residual, np.r_[sweep_residual, row_residual / 2])
    return coef


def _robust_weight(residual, reference):
    scale = max(1e-8, 4 * np.median(reference))
    return 1 / (1 + (residual / scale) ** 2)


def _forward(x, coef):
    b, a, s, c, q = coef.T
    v = x[:, _PAIR]
    return b + a*x + s*np.sin(_OWN_FREQ*x) + c*v + q*(1 - np.cos(_PAIR_FREQ*v))


def _inverse(raw, coef):
    shape = np.shape(raw)
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    b, a, s, c, q = coef.T
    x = (y - b) / a
    for _ in range(32):
        residual = _forward(x, coef) - y
        error = np.max(np.abs(residual), axis=1)
        active = error > 1e-11
        if not np.any(active):
            return x.reshape(shape)
        u, target, r = x[active], y[active], residual[active]
        diag = a + (_OWN_FREQ*s)*np.cos(_OWN_FREQ*u)
        off = c + (_PAIR_FREQ*q)*np.sin(_PAIR_FREQ*u[:, _PAIR])
        det = diag*diag[:, _PAIR] - off*off[:, _PAIR]
        step = (diag[:, _PAIR]*r - off*r[:, _PAIR]) / det
        candidate = u - step
        norm = np.sum(r*r, axis=1)
        for _ in range(24):
            trial = _forward(candidate, coef) - target
            bad = np.sum(trial*trial, axis=1) >= norm
            if not np.any(bad):
                break
            step[bad] *= 0.5
            candidate[bad] = u[bad] - step[bad]
        x[active] = candidate
    raise RuntimeError('Load-cell bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return calibrated histories and identity calibration without mutating the input."""
    calibration = dataset['calibration_load_cells']
    coef = _fit(calibration)
    result = dict(dataset)
    for key in ('train', 'test', 'chamber_verification_trials'):
        split = dataset.get(key)
        if split is not None:
            result[key] = {**split, **{field: _inverse(split[field], coef)
                                       for field in _HISTORY_FIELDS if field in split}}
    sweeps = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = {
        **calibration,
        'bridge_raw': calibration['true_channel_loads'].copy(),
        'single_channel_sweeps': {**sweeps, 'cmd_stress': sweeps['true_stress'].copy()},
    }
    return result
