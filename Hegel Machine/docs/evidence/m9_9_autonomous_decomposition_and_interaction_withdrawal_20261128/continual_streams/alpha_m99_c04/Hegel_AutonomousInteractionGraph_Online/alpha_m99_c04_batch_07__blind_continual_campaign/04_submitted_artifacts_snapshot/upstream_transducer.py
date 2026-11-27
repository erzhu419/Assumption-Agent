"""Calibrate the four coupled bridge channels in load coordinates."""
from __future__ import annotations

import numpy as np


def _fit_bridge(loads, raw):
    # Each bridge has linear cross-talk, an own-channel cubic, and an
    # interaction with the next load channel (with wraparound).
    design = np.stack((np.ones_like(loads), *[np.broadcast_to(loads[:, i, None], loads.shape) for i in range(4)],
                       loads**3, loads * np.roll(loads, -1, axis=1)), axis=-1)

    def fit(weights):
        return np.column_stack([
            np.linalg.lstsq(design[:, i] * weights[:, None], raw[:, i] * weights, rcond=None)[0]
            for i in range(4)
        ])

    coeff = fit(np.ones(len(loads)))
    for floor in (1., .1, .01, .001):
        for _ in range(12):
            err = np.linalg.norm(np.einsum('nci,ic->nc', design, coeff) - raw, axis=1)
            coeff = fit(1 / np.sqrt(np.maximum(err, floor)))
    err = np.linalg.norm(np.einsum('nci,ic->nc', design, coeff) - raw, axis=1)
    median = np.median(err)
    scale = 1.4826 * np.median(np.abs(err - median))
    return fit((err <= median + 8 * max(scale, 1e-9)).astype(float))


def _forward(loads, coeff):
    offset, linear, cubic, mixed = coeff[0], coeff[1:5], coeff[5], coeff[6]
    return offset + loads @ linear + cubic * loads**3 + mixed * loads * np.roll(loads, -1, axis=1)


def _invert(raw, coeff):
    target = np.asarray(raw, dtype=float).reshape(-1, 4)
    linear, cubic, mixed = coeff[1:5], coeff[5], coeff[6]
    x = np.linalg.solve(linear.T, (target - coeff[0]).T).T
    channels = np.arange(4)
    for _ in range(30):
        residual = _forward(x, coeff) - target
        if np.max(np.abs(residual), initial=0.) < 1e-11:
            break
        jac = np.broadcast_to(linear.T, (len(x), 4, 4)).copy()
        jac[:, channels, channels] += 3 * cubic * x**2 + mixed * np.roll(x, -1, axis=1)
        jac[:, channels, (channels + 1) % 4] += mixed * x
        delta = np.linalg.solve(jac, residual[..., None])[..., 0]
        trial = x - delta
        norm = np.linalg.norm(residual, axis=1)
        for _ in range(12):
            worse = np.linalg.norm(_forward(trial, coeff) - target, axis=1) > norm
            if not np.any(worse):
                break
            delta[worse] *= .5
            trial[worse] = x[worse] - delta[worse]
        x = trial
    return x.reshape(np.shape(raw))


def calibrate_upstream(dataset: dict) -> dict:
    """Return a copy of the dataset with canonical load coordinates."""
    cal = dataset['calibration_load_cells']
    coeff = _fit_bridge(np.asarray(cal['true_channel_loads']), np.asarray(cal['bridge_raw']))
    out = dict(dataset)
    sweeps = cal['single_channel_sweeps']
    out['calibration_load_cells'] = dict(cal, bridge_raw=cal['true_channel_loads'].copy(),
                                         single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()))
    for name in ('train', 'test', 'chamber_verification_trials'):
        group = dataset.get(name)
        if group is not None:
            out[name] = dict(group)
            for field in ('past_channel_raw', 'hold_channel_raw'):
                out[name][field] = _invert(group[field], coeff)
    return out
