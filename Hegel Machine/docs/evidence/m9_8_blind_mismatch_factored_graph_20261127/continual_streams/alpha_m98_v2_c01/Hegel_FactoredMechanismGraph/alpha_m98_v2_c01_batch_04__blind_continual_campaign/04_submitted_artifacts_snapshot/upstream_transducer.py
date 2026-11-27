"""Recover physical channel loads from the nonlinear load bridge."""
from itertools import combinations

import numpy as np

_PAIR_I, _PAIR_J = np.array(list(combinations(range(4), 2))).T


def _design(loads):
    return np.column_stack((np.ones(len(loads)), loads, loads**3,
                            loads[:, _PAIR_I] * loads[:, _PAIR_J]))


def _fit(loads, bridge):
    design = _design(loads)
    coeff = np.linalg.lstsq(design, bridge, rcond=None)[0]
    # A spike contaminates the entire observation, so weight rows, not cells.
    for step in range(40):
        residual = np.linalg.norm(design @ coeff - bridge, axis=1)
        scale = max(np.median(residual) * (1.5 if step < 10 else 0.8), 1e-9)
        weight = 1.0 / np.hypot(1.0, residual / scale)
        update = np.linalg.lstsq(design * weight[:, None],
                                 bridge * weight[:, None], rcond=None)[0]
        change = np.max(np.abs(update - coeff))
        coeff = update
        if step >= 10 and change < 1e-10:
            break
    residual = np.linalg.norm(design @ coeff - bridge, axis=1)
    clean = residual <= max(8.0 * np.median(residual), 1e-8)
    return np.linalg.lstsq(design[clean], bridge[clean], rcond=None)[0]


def _inverse(raw, coeff, lower, upper):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    linear, cubic, mixed = coeff[1:5], coeff[5:9], coeff[9:]
    loads = np.linalg.solve(linear.T, (target - coeff[0]).T).T
    # Only the starting guess is bounded; the inferred loads are not clipped.
    loads = np.clip(loads, lower, upper)
    for _ in range(30):
        residual = _design(loads) @ coeff - target
        error = np.max(np.abs(residual), axis=1)
        active = error > 1e-11
        if not np.any(active):
            break
        x = loads[active]
        jac = linear + 3.0 * x[:, :, None]**2 * cubic
        for i, j, term in zip(_PAIR_I, _PAIR_J, mixed):
            jac[:, i] += x[:, j, None] * term
            jac[:, j] += x[:, i, None] * term
        delta = np.linalg.solve(jac.swapaxes(1, 2), residual[active, :, None])[..., 0]
        trial = x - delta
        for _ in range(12):
            worse = np.max(np.abs(_design(trial) @ coeff - target[active]), axis=1) > error[active]
            if not np.any(worse):
                break
            delta[worse] *= 0.5
            trial[worse] = x[worse] - delta[worse]
        loads[active] = trial
    return loads.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a canonical-coordinate dataset without altering the input."""
    calibration = dataset['calibration_load_cells']
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    coeff = _fit(loads, np.asarray(calibration['bridge_raw'], dtype=float))
    lower, upper = loads.min(axis=0), loads.max(axis=0)
    result = dataset.copy()
    for key in ('train', 'test', 'chamber_verification_trials'):
        group = dataset.get(key)
        if group is not None:
            result[key] = group.copy()
            for field in ('past_channel_raw', 'hold_channel_raw'):
                result[key][field] = _inverse(group[field], coeff, lower, upper)
    sweeps = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        calibration, bridge_raw=calibration['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()))
    return result
