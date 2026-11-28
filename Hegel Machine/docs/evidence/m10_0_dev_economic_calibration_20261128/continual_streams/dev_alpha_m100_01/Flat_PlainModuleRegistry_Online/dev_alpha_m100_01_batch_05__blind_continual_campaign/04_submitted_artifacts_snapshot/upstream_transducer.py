"""Calibrate the paired four-channel bridge before constitutive fitting."""

import numpy as np
from scipy.optimize import least_squares

_PAIR = np.array([2, 3, 0, 1])
_LOAD_KEYS = ('past_channel_raw', 'hold_channel_raw')


def _clean_rows(loads, raw):
    """Use row residuals so a spike on one bridge cannot bias its pair."""
    design = np.column_stack((np.ones(len(loads)), loads))
    weight = np.ones(len(loads))
    for _ in range(12):
        coeff = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(design @ coeff - raw, axis=1)
        scale = max(np.median(error), 1e-10)
        weight = 1.0 / np.hypot(1.0, error / scale)
    return error <= 6.0 * scale


def _design(loads, freq):
    """One design matrix per channel; self and paired frequencies are distinct."""
    paired = loads[:, _PAIR]
    return np.stack((np.ones_like(loads), loads, np.sin(freq[0] * loads),
                     paired, -2.0 * np.sin(0.5 * freq[1] * paired)**2), axis=-1)


def _profile(loads, raw):
    def linear_fit(freq):
        design = _design(loads, freq)
        coeff = np.stack([np.linalg.lstsq(design[:, j], raw[:, j], rcond=None)[0]
                          for j in range(4)])
        return np.einsum('ncp,cp->nc', design, coeff) - raw, coeff

    freq = least_squares(lambda f: linear_fit(f)[0].ravel(), (2.4, 2.1),
                         bounds=(0.6, 5.0)).x
    return freq, linear_fit(freq)[1]


def _fit_bridge(loads, raw):
    clean = _clean_rows(loads, raw)
    freq, coeff = _profile(loads[clean], raw[clean])
    error = np.linalg.norm(_forward(loads[clean], freq, coeff) - raw[clean], axis=1)
    keep = error <= 6.0 * max(np.median(error), 1e-10)
    if not np.all(keep):
        freq, coeff = _profile(loads[clean][keep], raw[clean][keep])
    return freq, coeff


def _forward(loads, freq, coeff):
    b, a, s, c, d = coeff.T
    paired = loads[:, _PAIR]
    return b + a * loads + s * np.sin(freq[0] * loads) + c * paired - 2.0 * d * np.sin(0.5 * freq[1] * paired)**2


def _invert_bridge(raw, freq, coeff):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    b, a, s, c, d = coeff.T
    z = (y - b) / a
    # A Newton step is fast in the operating range. Use the contractive
    # fixed-point step whenever Newton does not improve the residual.
    contraction = (abs(s * freq[0]) + abs(c) + abs(d * freq[1])) / a
    if np.any(a <= 0) or np.any(contraction >= 1):
        raise ValueError('Fitted bridge is not diagonally dominant')
    for _ in range(600):
        residual = (_forward(z, freq, coeff) - y) / a
        error = np.max(np.abs(residual), axis=1)
        if np.all(error <= 1e-12):
            return z.reshape(raw.shape)
        diagonal = 1 + s * freq[0] / a * np.cos(freq[0] * z)
        cross = (c - d * freq[1] * np.sin(freq[1] * z[:, _PAIR])) / a
        det = diagonal * diagonal[:, _PAIR] - cross * cross[:, _PAIR]
        step = (diagonal[:, _PAIR] * residual - cross * residual[:, _PAIR]) / det
        trial = z - step
        trial_error = np.max(np.abs((_forward(trial, freq, coeff) - y) / a), axis=1)
        z = np.where((trial_error < 0.5 * error)[:, None], trial, z - residual)
    raise RuntimeError('Bridge inverse did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return canonical load coordinates without changing the input dataset."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    freq, coeff = _fit_bridge(loads, raw)
    result = dict(dataset)
    for name in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(name)
        if part is not None:
            result[name] = dict(part)
            for key in _LOAD_KEYS:
                result[name][key] = _invert_bridge(part[key], freq, coeff)

    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=np.asarray(cal['true_channel_loads']).copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy()),
    )
    return result
