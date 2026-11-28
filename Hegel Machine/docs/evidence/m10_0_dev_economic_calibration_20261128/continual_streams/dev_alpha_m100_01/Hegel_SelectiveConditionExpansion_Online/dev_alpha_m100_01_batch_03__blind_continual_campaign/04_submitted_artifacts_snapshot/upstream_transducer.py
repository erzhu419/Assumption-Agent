"""Calibrate the four-channel bridge before constitutive fitting."""
from __future__ import annotations

import numpy as np


_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4


def _design(load):
    shared = np.column_stack((np.ones(len(load)), load))
    pair = load * load[:, _NEXT]
    cube = load ** 3
    return [np.column_stack((shared, pair[:, i], cube[:, i])) for i in _CHANNEL]


def _fit(load, raw):
    design = _design(load)
    weights = np.ones(len(load))
    # Weight whole rows: a bridge spike can affect multiple outputs together.
    for _ in range(20):
        coef = np.stack([np.linalg.lstsq(a * weights[:, None], y * weights,
                                         rcond=None)[0]
                         for a, y in zip(design, raw.T)])
        err = np.linalg.norm(np.column_stack([a @ b for a, b in zip(design, coef)])
                             - raw, axis=1)
        scale = max(1e-8, np.median(err))
        weights = 1.0 / np.hypot(1.0, err / (2.0 * scale))
    clean = err < 8.0 * scale
    return np.stack([np.linalg.lstsq(a[clean], y[clean], rcond=None)[0]
                     for a, y in zip(design, raw.T)])


def _forward(load, coef):
    return (coef[:, 0] + load @ coef[:, 1:5].T
            + coef[:, 5] * load * load[:, _NEXT] + coef[:, 6] * load ** 3)


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    offset, linear = coef[:, 0], coef[:, 1:5]
    pair, cube = coef[:, 5], coef[:, 6]
    load = np.linalg.solve(linear, (y - offset).T).T
    for _ in range(40):
        residual = _forward(load, coef) - y
        error = np.max(np.abs(residual), axis=1)
        active = error > 1e-11
        if not np.any(active):
            return load.reshape(raw.shape)
        z = load[active]
        jac = np.broadcast_to(linear, (len(z), 4, 4)).copy()
        jac[:, _CHANNEL, _CHANNEL] += pair * z[:, _NEXT] + 3 * cube * z ** 2
        jac[:, _CHANNEL, _NEXT] += pair * z
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        trial = z - step
        # If the affine starting point overshoots, shorten only those steps.
        target, previous = y[active], error[active]
        for _ in range(12):
            worse = np.max(np.abs(_forward(trial, coef) - target), axis=1) > previous
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        load[active] = trial
    if np.max(np.abs(_forward(load, coef) - y)) > 1e-9:
        raise RuntimeError('Bridge inverse did not converge')
    return load.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return trial channels and calibration references in true-load coordinates."""
    cal = dataset['calibration_load_cells']
    load = np.asarray(cal['true_channel_loads'])
    raw = np.asarray(cal['bridge_raw'])
    result = dict(dataset)
    normalized = dict(cal, bridge_raw=load.copy())
    sweeps = cal.get('single_channel_sweeps')
    if sweeps is not None:
        normalized['single_channel_sweeps'] = dict(
            sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy())
    result['calibration_load_cells'] = normalized
    if not np.array_equal(load, raw):
        coef = _fit(load, raw)
        for name in ('train', 'test', 'chamber_verification_trials'):
            part = dataset.get(name)
            if part is not None:
                result[name] = dict(part, **{
                    k: _inverse(part[k], coef)
                    for k in ('past_channel_raw', 'hold_channel_raw')
                })
    return result
