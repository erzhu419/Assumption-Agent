"""Fit and invert the four-channel chamber bridge in load coordinates."""

import numpy as np
from scipy.optimize import least_squares

_PARTNER = np.array([2, 3, 0, 1])
_START = np.array([2.5, 2.0])


def _basis(load, channel, frequency):
    own = load[:, channel]
    other = load[:, _PARTNER[channel]]
    return np.column_stack((np.ones(len(load)), load,
                            np.sin(frequency[0] * own),
                            np.cos(frequency[1] * other) - 1))


def _fit(load, raw):
    # Start with the identified bridge basis. Rowwise Cauchy weights suppress
    # simultaneous spikes in any subset of the bridge outputs.
    bases = [_basis(load, j, _START) for j in range(4)]
    weight = np.ones(len(load))
    for _ in range(18):
        coef = np.stack([np.linalg.lstsq(a * weight[:, None], raw[:, j] * weight,
                                        rcond=None)[0] for j, a in enumerate(bases)])
        residual = np.linalg.norm(np.column_stack([a @ b for a, b in zip(bases, coef)])
                                  - raw, axis=1)
        scale = max(2 * np.median(residual), 1e-8)
        weight = 1 / np.hypot(1, residual / scale)
    mid = np.median(residual)
    keep = residual <= mid + max(8 * np.median(np.abs(residual - mid)), 3 * mid, 1e-8)
    x, y = load[keep], raw[keep]

    # Profile the linear gains and offsets out of the frequency fit.
    def profile(frequency, return_coef=False):
        a = [_basis(x, j, frequency) for j in range(4)]
        b = np.stack([np.linalg.lstsq(m, y[:, j], rcond=None)[0]
                      for j, m in enumerate(a)])
        if return_coef:
            return b
        return (np.column_stack([m @ v for m, v in zip(a, b)]) - y).ravel()

    frequency = least_squares(profile, _START, bounds=([1.5, 1.0], [3.5, 3.0])).x
    return profile(frequency, True), frequency


def _forward(load, coef, frequency):
    offset, linear, sine, cosine = coef[:, 0], coef[:, 1:5], coef[:, 5], coef[:, 6]
    return (load @ linear.T + offset + sine * np.sin(frequency[0] * load)
            + cosine * (np.cos(frequency[1] * load[:, _PARTNER]) - 1))


def _jacobian(load, coef, frequency):
    i = np.arange(4)
    jac = np.broadcast_to(coef[:, 1:5], (len(load), 4, 4)).copy()
    jac[:, i, i] += coef[:, 5] * frequency[0] * np.cos(frequency[0] * load)
    jac[:, i, _PARTNER] -= (coef[:, 6] * frequency[1]
                            * np.sin(frequency[1] * load[:, _PARTNER]))
    return jac


def _inverse(raw, coef, frequency):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    load = np.linalg.solve(coef[:, 1:5], (target - coef[:, 0]).T).T.copy()
    # Damped Newton iteration, with line search independent for each trial row.
    for _ in range(24):
        diff = _forward(load, coef, frequency) - target
        active = np.flatnonzero(np.max(np.abs(diff), axis=1) > 1e-11)
        if not active.size:
            break
        step = np.linalg.solve(_jacobian(load[active], coef, frequency),
                               diff[active, :, None])[..., 0]
        error = np.sum(diff[active] ** 2, axis=1)
        for _ in range(14):
            trial = load[active] - step
            new_error = np.sum((_forward(trial, coef, frequency) - target[active]) ** 2, axis=1)
            better = new_error < error
            load[active[better]] = trial[better]
            active, step, error = active[~better], step[~better] * 0.5, error[~better]
            if not active.size:
                break
    return load.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a canonical-coordinate copy without changing the input dataset."""
    cal = dataset['calibration_load_cells']
    true_load = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    result = dict(dataset)
    if not np.array_equal(raw, true_load):
        coef, frequency = _fit(true_load, raw)
        for key in ('train', 'test', 'chamber_verification_trials'):
            block = dataset.get(key)
            if block is not None:
                result[key] = dict(block)
                for field in ('past_channel_raw', 'hold_channel_raw'):
                    result[key][field] = _inverse(block[field], coef, frequency)
    result['calibration_load_cells'] = dict(cal, bridge_raw=cal['true_channel_loads'].copy())
    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells']['single_channel_sweeps'] = dict(
        sweeps, cmd_stress=sweeps['true_stress'].copy())
    return result
