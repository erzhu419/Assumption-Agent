"""Fit and invert the coupled load-cell bridge before constitutive fitting."""
from itertools import combinations

import numpy as np

_PAIRS = tuple(combinations(range(4), 2))
_RAW_KEYS = ('past_channel_raw', 'hold_channel_raw')


def _basis(load):
    return np.column_stack((np.ones(len(load)), load,
                            *(load[:, i] * load[:, j] for i, j in _PAIRS),
                            load ** 3))


def _fit(load, raw):
    a = _basis(load)
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Rowwise Cauchy weighting: a single spike can hit several bridge channels.
    for _ in range(35):
        err = np.linalg.norm(a @ coef - raw, axis=1)
        scale = max(2 * np.median(err), 1e-9)
        w = 1 / np.hypot(1, err / scale)
        coef = np.linalg.lstsq(a * w[:, None], raw * w[:, None], rcond=None)[0]
    err = np.linalg.norm(a @ coef - raw, axis=1)
    clean = err <= max(8 * np.median(err), 1e-8)
    a, raw = a[clean], raw[clean]
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Refit only resolved terms; otherwise noise in unused powers grows
    # when operational loads extend beyond the calibration envelope.
    variance = np.sum((a @ coef - raw) ** 2, axis=0) / max(len(a) - a.shape[1], 1)
    sensitivity = np.linalg.norm(np.linalg.pinv(a), axis=1)
    for channel in range(4):
        keep = np.abs(coef[:, channel]) > 5 * np.sqrt(variance[channel]) * sensitivity
        keep[:5] = True
        coef[:, channel] = 0
        coef[keep, channel] = np.linalg.lstsq(a[:, keep], raw[:, channel], rcond=None)[0]
    return coef


def _jacobian(load, coef):
    # Axes are (sample, bridge output, applied load).
    jac = np.broadcast_to(coef[1:5].T, (len(load), 4, 4)).copy()
    for term, (i, j) in enumerate(_PAIRS, 5):
        jac[:, :, i] += load[:, j, None] * coef[term]
        jac[:, :, j] += load[:, i, None] * coef[term]
    jac += 3 * load[:, None, :] ** 2 * coef[11:].T
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    load = np.linalg.solve(coef[1:5].T, (target - coef[0]).T).T
    for _ in range(30):
        residual = _basis(load) @ coef - target
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-11
        if not active.any():
            break
        x, y = load[active], target[active]
        step = np.linalg.solve(_jacobian(x, coef), residual[active, :, None])[..., 0]
        trial = x - step
        old = error[active]
        for _ in range(16):
            worse = np.linalg.norm(_basis(trial) @ coef - y, axis=1) >= old
            if not worse.any():
                break
            step[worse] *= 0.5
            trial[worse] = x[worse] - step[worse]
        load[active] = trial
    return load.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a calibrated copy of the dataset without modifying its arrays."""
    cal = dataset['calibration_load_cells']
    coef = _fit(cal['true_channel_loads'], cal['bridge_raw'])
    result = dict(dataset)
    for name in ('train', 'test', 'chamber_verification_trials'):
        split = dataset.get(name)
        if split is not None:
            result[name] = {**split, **{key: _invert(split[key], coef) for key in _RAW_KEYS}}
    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells'] = {
        **cal,
        'bridge_raw': cal['true_channel_loads'].copy(),
        'single_channel_sweeps': {**sweeps, 'cmd_stress': sweeps['true_stress'].copy()},
    }
    return result
