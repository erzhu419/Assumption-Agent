"""Recover applied channel loads from the coupled load-cell bridge."""
from itertools import combinations_with_replacement

import numpy as np


_I, _J = np.array(list(combinations_with_replacement(range(4), 2))).T


def _basis(load):
    return np.column_stack((np.ones(len(load)), load,
                            load[:, _I] * load[:, _J], load**3))


def _fit_bridge(load, raw):
    design = _basis(load)
    weight = np.ones(len(load))
    floor = 1e-12 * max(1.0, np.median(np.abs(raw)))
    # Start with a robust row-norm fit before stronger spike rejection.
    # A least-squares start can mask a spike at a high-leverage load.
    for _ in range(32):
        coef = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None],
                               rcond=None)[0]
        error = np.linalg.norm(design @ coef - raw, axis=1)
        weight = 1.0 / np.sqrt(np.hypot(error, floor))

    # Treat a spike as contamination of the whole calibration row.
    for _ in range(16):
        coef = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None],
                               rcond=None)[0]
        error = np.linalg.norm(design @ coef - raw, axis=1)
        scale = max(0.3 * np.median(error), floor)
        weight = 1.0 / np.hypot(1.0, error / scale)
    clean = error <= max(6.0 * np.median(error), floor)
    return np.linalg.lstsq(design[clean], raw[clean], rcond=None)[0]


def _jacobian(load, coef):
    # Axes are sample, bridge output, applied-load coordinate.
    jac = np.broadcast_to(coef[1:5].T, (len(load), 4, 4)).copy()
    jac += (3.0 * load[:, None, :]**2) * coef[15:19].T
    for i, j, quadratic in zip(_I, _J, coef[5:15]):
        jac[:, :, i] += load[:, j, None] * quadratic
        jac[:, :, j] += load[:, i, None] * quadratic
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    load = np.linalg.solve(coef[1:5].T, (target - coef[0]).T).T
    for _ in range(24):
        residual = _basis(load) @ coef - target
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-11 * (1.0 + np.linalg.norm(target, axis=1))
        if not np.any(active):
            break
        current = load[active]
        goal = target[active]
        step = np.linalg.solve(_jacobian(current, coef), residual[active, :, None])[:, :, 0]
        trial = current - step
        # Backtrack independently: a difficult load must not slow other rows.
        for _ in range(12):
            worse = np.linalg.norm(_basis(trial) @ coef - goal, axis=1) > error[active]
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = current[worse] - step[worse]
        load[active] = trial
    return load.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return the dataset in applied-load coordinates without changing the input."""
    cells = dataset['calibration_load_cells']
    loads = np.asarray(cells['true_channel_loads'], dtype=float)
    coef = _fit_bridge(loads, np.asarray(cells['bridge_raw'], dtype=float))

    sweeps = cells['single_channel_sweeps']
    calibrated = dict(dataset)
    calibrated['calibration_load_cells'] = dict(
        cells, bridge_raw=cells['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    for name in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(name)
        if block is not None:
            calibrated[name] = dict(block, **{
                key: _invert(block[key], coef)
                for key in ('past_channel_raw', 'hold_channel_raw')
            })
    return calibrated
