"""Recover channel loads from a coupled nonlinear bridge calibration."""

import numpy as np


_PAIRS = tuple((i, j) for i in range(4) for j in range(i, 4))
_CHANNEL_FIELDS = ('past_channel_raw', 'hold_channel_raw')


def _design(x):
    return np.column_stack((np.ones(len(x)), x,
                            *[x[:, i] * x[:, j] for i, j in _PAIRS], x**3))


def _fit(loads, raw):
    design = _design(loads)
    coef = np.linalg.lstsq(design, raw, rcond=None)[0]
    for _ in range(15):
        residual = np.linalg.norm(raw - design @ coef, axis=1)
        scale = max(2 * np.median(residual), 1e-5)
        weight = 1 / np.hypot(1, residual / scale)
        coef = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None],
                               rcond=None)[0]
    return coef


def _jacobian(x, coef):
    # Axes are sample, raw output, and true-load input, as in J dx = residual.
    jac = np.broadcast_to(coef[1:5].T, (len(x), 4, 4)).copy()
    for b, (i, j) in zip(coef[5:-4], _PAIRS):
        jac[:, :, i] += x[:, j, None] * b
        jac[:, :, j] += x[:, i, None] * b
    jac += 3 * x[:, None, :]**2 * coef[-4:].T
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    if not len(y):
        return raw.copy()
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    for _ in range(20):
        error = _design(x) @ coef - y
        if np.max(np.abs(error)) < 1e-11:
            break
        step = np.linalg.solve(_jacobian(x, coef), error[..., None])[..., 0]
        proposal = x - step
        loss = np.sum(error**2, axis=1)
        # Set each row's step independently so one difficult load cannot slow the rest.
        for _ in range(12):
            worse = np.sum((_design(proposal) @ coef - y)**2, axis=1) > loss
            if not np.any(worse):
                break
            step[worse] *= 0.5
            proposal[worse] = x[worse] - step[worse]
        x = proposal
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a dataset with bridge readings replaced by canonical loads."""
    calibration = dataset['calibration_load_cells']
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    coef = _fit(loads, np.asarray(calibration['bridge_raw'], dtype=float))
    output = dict(dataset)
    output['calibration_load_cells'] = dict(calibration, bridge_raw=loads.copy())
    sweeps = calibration['single_channel_sweeps']
    output['calibration_load_cells']['single_channel_sweeps'] = dict(
        sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy())
    for name in ('train', 'test', 'chamber_verification_trials'):
        if dataset.get(name) is not None:
            split = dict(dataset[name])
            for field in _CHANNEL_FIELDS:
                split[field] = _invert(split[field], coef)
            output[name] = split
    return output
