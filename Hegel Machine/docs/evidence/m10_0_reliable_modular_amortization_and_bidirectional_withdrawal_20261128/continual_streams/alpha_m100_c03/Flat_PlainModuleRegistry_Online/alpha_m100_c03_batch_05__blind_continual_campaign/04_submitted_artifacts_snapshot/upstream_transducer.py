"""Fit and invert the paired four-channel load-cell bridge."""

import numpy as np
from scipy.optimize import least_squares


_PAIR = np.array([2, 3, 0, 1])


def _basis(x, frequencies):
    own, cross = frequencies
    paired = x[:, _PAIR]
    return np.stack((np.ones_like(x), x, np.sin(own * x), paired,
                     1 - np.cos(cross * paired)), axis=1).transpose(2, 0, 1)


def _fit_bridge(load, raw):
    # Screen gross readout spikes before profiling frequencies, then reject
    # any remaining row spikes against the fitted multichannel response.
    centered = np.abs(raw - np.median(raw, axis=0))
    spread = np.median(centered, axis=0)
    keep = np.all(centered <= 12 * np.maximum(spread, 0.05), axis=1)
    frequency = np.array([2.5, 2.0])
    for _ in range(5):
        x, y = load[keep], raw[keep]

        def profile(f):
            design = _basis(x, f)
            coeff = np.stack([np.linalg.lstsq(a, b, rcond=None)[0]
                              for a, b in zip(design, y.T)])
            return np.einsum('cnk,ck->nc', design, coeff) - y, coeff

        frequency = least_squares(lambda f: profile(f)[0].ravel(), frequency,
                                  bounds=([1.8, 0.9], [3.2, 3.5]),
                                  max_nfev=60).x
        _, coeff = profile(frequency)
        error = _forward(load, frequency, coeff) - raw
        centered = np.abs(error - np.median(error[keep], axis=0))
        spread = 1.4826 * np.median(centered[keep], axis=0)
        updated = np.all(centered <= 6 * np.maximum(spread, 2e-4), axis=1)
        if np.array_equal(updated, keep):
            break
        keep = updated
    return frequency, coeff


def _forward(x, frequency, coeff):
    bias, linear, sine, coupling, cosine = coeff.T
    own, cross = frequency
    p = x[:, _PAIR]
    return (bias + linear * x + sine * np.sin(own * x) + coupling * p
            + cosine * (1 - np.cos(cross * p)))


def _invert(raw, frequency, coeff):
    shape = raw.shape
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    if not len(y):
        return y.reshape(shape).copy()
    bias, linear, sine, coupling, cosine = coeff.T
    own, cross = frequency
    linear_map = np.diag(linear)
    linear_map[np.arange(4), _PAIR] = coupling
    x = np.linalg.solve(linear_map, (y - bias).T).T
    pairs = np.array([[0, 2], [1, 3]])
    for _ in range(40):
        error = _forward(x, frequency, coeff) - y
        loss = np.sum(error[:, pairs] ** 2, axis=-1)
        if np.max(loss) < 1e-22:
            break
        diagonal = linear + (sine * own) * np.cos(own * x)
        offdiag = coupling + (cosine * cross) * np.sin(cross * x[:, _PAIR])
        determinant = diagonal * diagonal[:, _PAIR] - offdiag * offdiag[:, _PAIR]
        delta = (diagonal[:, _PAIR] * error - offdiag * error[:, _PAIR]) / determinant
        step = np.ones_like(loss)
        for _ in range(20):
            trial = x - step[:, [0, 1, 0, 1]] * delta
            residual = _forward(trial, frequency, coeff) - y
            improved = np.sum(residual[:, pairs] ** 2, axis=-1) < loss
            pending = (loss >= 1e-22) & ~improved
            if not np.any(pending):
                break
            step[pending] *= 0.5
        x = np.where(improved[:, [0, 1, 0, 1]], trial, x)
    return x.reshape(shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return calibrated load histories without altering the input dataset."""
    calibration = dataset['calibration_load_cells']
    load = np.asarray(calibration['true_channel_loads'], dtype=float)
    frequency, coeff = _fit_bridge(load, np.asarray(calibration['bridge_raw'], dtype=float))
    normalized = dict(calibration, bridge_raw=calibration['true_channel_loads'].copy())
    sweeps = calibration.get('single_channel_sweeps')
    if sweeps is not None:
        normalized['single_channel_sweeps'] = dict(sweeps, cmd_stress=sweeps['true_stress'].copy())
    result = dict(dataset, calibration_load_cells=normalized)
    for name in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(name)
        if block is not None:
            result[name] = dict(block)
            for field in ('past_channel_raw', 'hold_channel_raw'):
                result[name][field] = _invert(block[field], frequency, coeff)
    return result
