"""Fit and invert the coupled four-channel load bridge."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

_PAIR = np.array([2, 3, 0, 1])


def _fit_bridge(calibration):
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    sweeps = calibration['single_channel_sweeps']
    models = []
    for j, k in enumerate(_PAIR):
        m = np.asarray(sweeps['lab_id']) == j
        u = np.r_[loads[:, j], np.asarray(sweeps['true_stress'])[m]]
        v = np.r_[loads[:, k], np.zeros(m.sum())]
        y = np.r_[raw[:, j], np.asarray(sweeps['cmd_stress'])[m]]

        def design(freq):
            return np.column_stack((np.ones_like(u), u, np.sin(freq[0] * u),
                                    v, np.sin(freq[1] * v)**2))

        start = np.array([2.5, 1.0])
        # Start from central readings so additive spikes cannot set the initial gains.
        dev = np.abs(y - np.median(y))
        keep = dev < 6 * max(np.median(dev), .1)
        a = design(start)
        coef = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]

        def residual(p):
            return design(p[-2:]) @ p[:5] - y

        bounds = ([-np.inf]*5 + [1.2, .45], [np.inf]*5 + [4.0, 1.6])
        p = np.r_[coef, start]
        for scale, loss in ((.10, 'soft_l1'), (.01, 'cauchy'), (.001, 'cauchy')):
            p = least_squares(residual, p, bounds=bounds, loss=loss,
                              f_scale=scale, max_nfev=160).x
        models.append(p)
    return np.array(models)


def _forward(x, model):
    bias, gain, amp, cross, bend, freq, freq_cross = model.T
    v = x[..., _PAIR]
    return bias + gain*x + amp*np.sin(freq*x) + cross*v + bend*np.sin(freq_cross*v)**2


def _invert(raw, model):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    bias, gain, amp, cross, bend, freq, freq_cross = model.T
    x = (y - bias) / gain
    for _ in range(45):
        res = _forward(x, model) - y
        err = np.max(np.abs(res), axis=1, initial=0.)
        if np.max(err, initial=0.) < 1e-11:
            break
        a = gain + amp*freq*np.cos(freq*x)
        b = cross + bend*freq_cross*np.sin(2*freq_cross*x[:, _PAIR])
        det = a*a[:, _PAIR] - b*b[:, _PAIR]
        step = (res*a[:, _PAIR] - b*res[:, _PAIR]) / det
        step /= np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
        scale = np.ones((len(x), 1))
        z = x - step
        for _ in range(16):
            bad = np.max(np.abs(_forward(z, model) - y), axis=1) > err
            if not np.any(bad):
                break
            scale[bad] *= .5
            z[bad] = x[bad] - scale[bad]*step[bad]
        x = z
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return the same dataset in true-load coordinates without mutating it."""
    calibration = dataset['calibration_load_cells']
    model = _fit_bridge(calibration)
    result = dataset.copy()
    sweeps = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        calibration, bridge_raw=calibration['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()))
    for section in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(section)
        if part is not None:
            result[section] = dict(part, **{
                key: _invert(part[key], model)
                for key in ('past_channel_raw', 'hold_channel_raw')
            })
    return result
