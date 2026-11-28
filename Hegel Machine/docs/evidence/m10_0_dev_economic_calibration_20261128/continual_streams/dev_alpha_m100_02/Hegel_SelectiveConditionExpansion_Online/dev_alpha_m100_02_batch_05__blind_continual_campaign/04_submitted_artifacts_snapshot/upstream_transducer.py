"""Fit and invert the coupled four-channel load bridge."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

_PAIR = np.array([2, 3, 0, 1])


def _fit_bridge(calibration):
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    sweeps = calibration['single_channel_sweeps']
    lab = np.asarray(sweeps['lab_id'], dtype=int)
    j = np.r_[np.tile(np.arange(4), len(loads)), lab]
    u = np.r_[loads.ravel(), sweeps['true_stress']]
    v = np.r_[loads[:, _PAIR].ravel(), np.zeros(len(lab))]
    y = np.r_[raw.ravel(), sweeps['cmd_stress']]

    # Reject gross calibration spikes before estimating shared response shape.
    dev = np.abs(raw - np.median(raw, axis=0))
    span = np.maximum(np.median(dev, axis=0), .35)
    good_rows = np.all(dev < 8 * span, axis=1)
    sy = np.asarray(sweeps['cmd_stress'])
    good_sweeps = np.ones(len(lab), dtype=bool)
    for q in range(4):
        mask = lab == q
        if np.any(mask):
            delta = np.abs(sy[mask] - np.median(sy[mask]))
            good_sweeps[mask] = delta < 8 * max(np.median(delta), .35)
    keep = np.r_[np.repeat(good_rows, 4), good_sweeps]

    def response(shape):
        amp, cross, bend, freq, freq_cross = shape
        return u + amp*np.sin(freq*u) + cross*v + bend*np.sin(freq_cross*v)**2

    def residual(p):
        return p[j] + p[4+j]*response(p[8:]) - y

    shape = [.2, -.2, .1, 2.5, 1.]
    z = response(shape)
    p = np.zeros(13)
    for q in range(4):
        mask = keep & (j == q)
        p[q], p[4+q] = np.linalg.lstsq(
            np.column_stack((np.ones(mask.sum()), z[mask])), y[mask], rcond=None)[0]
    p[8:] = shape
    for scale, loss in ((.05, 'soft_l1'), (.003, 'cauchy')):
        p = least_squares(lambda p: residual(p)[keep], p, loss=loss,
                          f_scale=scale, max_nfev=160).x
    # Trim remaining spikes at the row level and refit at measurement precision.
    r = np.abs(residual(p))
    tol = max(.002, 8*np.median(r[keep]))
    clean = np.r_[np.repeat(np.max(r[:raw.size].reshape(raw.shape), axis=1) < tol, 4),
                  r[raw.size:] < tol]
    p = least_squares(lambda p: residual(p)[clean], p, max_nfev=100).x
    amp, cross, bend, freq, freq_cross = p[8:]
    gain = p[4:8]
    return np.column_stack((p[:4], gain, amp*gain, cross*gain, bend*gain,
                            np.full(4, freq), np.full(4, freq_cross)))


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
