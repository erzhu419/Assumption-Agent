"""Calibrate the four-channel bridge before constitutive fitting."""
from __future__ import annotations

import numpy as np

_PARTNER = np.array([2, 3, 0, 1])
_SELF_FREQUENCY = 2.5
_CROSS_FREQUENCY = 2.0


def _design(load, paired):
    return np.stack((np.ones_like(load), load, np.sin(_SELF_FREQUENCY * load),
                     paired, 1 - np.cos(_CROSS_FREQUENCY * paired)), axis=-1)


def _fit_bridge(calibration):
    truth = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    sweep = calibration['single_channel_sweeps']
    axis_truth = np.asarray(sweep['true_stress'], dtype=float)
    axis_raw = np.asarray(sweep['cmd_stress'], dtype=float)
    axis_channel = np.asarray(sweep['lab_id'])
    mixed = _design(truth, truth[:, _PARTNER])
    axis = _design(axis_truth, np.zeros_like(axis_truth))
    keep_mixed = np.ones(len(truth), dtype=bool)
    keep_axis = np.ones(len(axis_truth), dtype=bool)

    def fit(weights, axis_weights):
        result = []
        for j in range(4):
            m = axis_channel == j
            rows = np.vstack((mixed[:, j], axis[m]))
            values = np.r_[raw[:, j], axis_raw[m]]
            w = np.sqrt(np.r_[weights, axis_weights[m]])
            result.append(np.linalg.lstsq(rows * w[:, None], values * w, rcond=None)[0])
        return np.array(result)

    # Downweight a spike as one observation, not four independent residuals.
    coef = fit(keep_mixed, keep_axis)
    for _ in range(8):
        e = np.linalg.norm(np.einsum('ncf,cf->nc', mixed, coef) - raw, axis=1) / 2
        ea = np.abs(np.sum(axis * coef[axis_channel], axis=1) - axis_raw)
        scale = max(np.median(np.r_[e, ea]), 1e-12)
        coef = fit(1 / (1 + (e / (3 * scale))**2),
                   1 / (1 + (ea / (3 * scale))**2))
    e = np.linalg.norm(np.einsum('ncf,cf->nc', mixed, coef) - raw, axis=1) / 2
    ea = np.abs(np.sum(axis * coef[axis_channel], axis=1) - axis_raw)
    cutoff = 8 * max(np.median(np.r_[e, ea]), 1e-12)
    return fit(e <= cutoff, ea <= cutoff)


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    bias, gain, sine, coupling, cosine = coef.T
    linear = np.diag(gain)
    linear[np.arange(4), _PARTNER] = coupling
    x = np.linalg.solve(linear, (target - bias).T).T

    def residual(x):
        y = x[:, _PARTNER]
        return (bias + gain*x + sine*np.sin(_SELF_FREQUENCY*x) + coupling*y
                + cosine*(1 - np.cos(_CROSS_FREQUENCY*y)) - target)

    r = residual(x)
    tol = 1e-11 * (1 + np.max(np.abs(target), axis=1))
    for _ in range(45):
        error = np.max(np.abs(r), axis=1)
        if np.all(error <= tol):
            return x.reshape(raw.shape)
        own = gain + _SELF_FREQUENCY*sine*np.cos(_SELF_FREQUENCY*x)
        other = coupling + _CROSS_FREQUENCY*cosine*np.sin(_CROSS_FREQUENCY*x[:, _PARTNER])
        det = own*own[:, _PARTNER] - other*other[:, _PARTNER]
        step = (own[:, _PARTNER]*r - other*r[:, _PARTNER]) / det
        step[error <= tol] = 0
        trial = x - step
        trial_r = residual(trial)
        rejected = np.max(np.abs(trial_r), axis=1) > error
        while np.any(rejected):
            step[rejected] *= 0.5
            trial[rejected] = x[rejected] - step[rejected]
            trial_r = residual(trial)
            rejected = np.max(np.abs(trial_r), axis=1) > error
        x, r = trial, trial_r
    raise RuntimeError('Bridge inverse did not converge')


def _calibrate_trials(trials, coef):
    if trials is None:
        return None
    if isinstance(trials, (list, tuple)):
        calibrated = [_calibrate_trials(trial, coef) for trial in trials]
        return tuple(calibrated) if isinstance(trials, tuple) else calibrated
    calibrated = dict(trials)
    for key, value in trials.items():
        if key in ('past_channel_raw', 'hold_channel_raw'):
            calibrated[key] = _invert(value, coef)
        elif isinstance(value, dict):
            calibrated[key] = _calibrate_trials(value, coef)
    return calibrated


def calibrate_upstream(dataset: dict) -> dict:
    """Return a load-calibrated dataset without modifying the input."""
    calibration = dataset['calibration_load_cells']
    coef = _fit_bridge(calibration)
    result = dict(dataset)
    for key in ('train', 'test', 'chamber_verification_trials'):
        if key in dataset:
            result[key] = _calibrate_trials(dataset[key], coef)
    sweep = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        calibration, bridge_raw=calibration['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweep, cmd_stress=sweep['true_stress'].copy()))
    return result
