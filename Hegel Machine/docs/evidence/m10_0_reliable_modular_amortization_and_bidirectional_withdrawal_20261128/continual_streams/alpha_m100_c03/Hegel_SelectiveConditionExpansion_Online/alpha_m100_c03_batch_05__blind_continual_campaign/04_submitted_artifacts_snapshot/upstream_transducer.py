"""Robust calibration of the coupled four-channel bridge."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


_OPPOSITE = [2, 3, 0, 1]


def _design(x, freq):
    opposite = x[:, _OPPOSITE]
    return np.stack((np.ones_like(x), x, opposite, np.sin(freq[0] * x),
                     np.cos(freq[1] * opposite) - 1), axis=-1)


def _coefficients(a, y, weights):
    w = np.sqrt(weights)
    return np.stack([np.linalg.lstsq(a[:, j] * w[:, None], y[:, j] * w,
                                     rcond=None)[0] for j in range(4)])


def _fit(x, y):
    # First isolate spikes with rowwise weighting: a spike in one bridge
    # reading makes the whole reference row suspect.
    freq = np.array([2.5, 2.0])
    a = _design(x, freq)
    weights = np.ones(len(x))
    for _ in range(12):
        p = _coefficients(a, y, weights)
        score = np.linalg.norm((a * p).sum(axis=-1) - y, axis=1)
        scale = max(2 * np.median(score), 1e-9)
        weights = 1 / (1 + (score / scale) ** 2)
    keep = score <= 6 * max(np.median(score), 1e-9)
    x, y = x[keep], y[keep]
    weights = np.ones(len(x))

    # The bridge channels share frequencies, but gains, offsets and coupling
    # strengths are fit independently. Profile those linear coefficients out.
    def profile(f, return_coefficients=False):
        a = _design(x, f)
        p = _coefficients(a, y, weights)
        return p if return_coefficients else ((a * p).sum(axis=-1) - y).ravel()

    freq = least_squares(profile, freq, bounds=([0.5, 0.5], [4.5, 4.5]),
                         max_nfev=60).x
    return freq, profile(freq, True)


def _forward(x, freq, p):
    return (_design(x, freq) * p).sum(axis=-1)


def _inverse(raw, freq, p):
    y = np.asarray(raw, dtype=float)
    flat = y.reshape(-1, 4)
    # Start from the coupled linear part. The Newton system consists of two
    # independent 2-by-2 systems (channels 0/2 and 1/3).
    i = np.arange(4)
    linear = np.diag(p[:, 1])
    linear[i, _OPPOSITE] = p[:, 2]
    x = np.linalg.solve(linear, (flat - p[:, 0]).T).T.copy()
    for _ in range(50):
        r = _forward(x, freq, p) - flat
        error = np.max(abs(r), axis=1)
        if np.all(error < 1e-11):
            return x.reshape(y.shape)
        diag = p[:, 1] + freq[0] * p[:, 3] * np.cos(freq[0] * x)
        cross = p[:, 2] - freq[1] * p[:, 4] * np.sin(freq[1] * x[:, _OPPOSITE])
        det = diag * diag[:, _OPPOSITE] - cross * cross[:, _OPPOSITE]
        step = (diag[:, _OPPOSITE] * r - cross * r[:, _OPPOSITE]) / det
        trial = x - step
        # Backtrack only for rows that do not improve on the full step.
        for _ in range(16):
            retry = (error >= 1e-11) & (np.max(abs(_forward(trial, freq, p) - flat), axis=1) >= error)
            if not retry.any():
                break
            step[retry] *= 0.5
            trial[retry] = x[retry] - step[retry]
        x = trial
    if np.any(abs(_forward(x, freq, p) - flat) > 1e-8):
        raise RuntimeError('Coupled bridge inversion did not converge')
    return x.reshape(y.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return the dataset in reference-load coordinates without mutating it."""
    cal = dataset['calibration_load_cells']
    true = np.asarray(cal['true_channel_loads'])
    raw = np.asarray(cal['bridge_raw'])
    model = None if np.array_equal(true, raw) else _fit(true, raw)

    def normalize(block):
        if isinstance(block, (list, tuple)):
            return type(block)(normalize(trial) for trial in block)
        result = dict(block)
        for key in ('past_channel_raw', 'hold_channel_raw'):
            if key in block:
                result[key] = np.array(block[key], copy=True) if model is None else _inverse(block[key], *model)
        return result

    out = dict(dataset)
    for key in ('train', 'test', 'chamber_verification_trials'):
        if dataset.get(key) is not None:
            out[key] = normalize(dataset[key])
    sweeps = cal['single_channel_sweeps']
    out['calibration_load_cells'] = dict(cal, bridge_raw=true.copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=np.array(sweeps['true_stress'], copy=True)))
    return out
