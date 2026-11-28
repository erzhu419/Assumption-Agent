"""Robust four-channel bridge calibration in the canonical load coordinates."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

_PAIR = np.array([2, 3, 0, 1])
_CHANNEL = np.arange(4)


def _response(x, shape):
    a, k, b, c, h = shape
    partner = x[..., _PAIR]
    return x + a * np.sin(k * x) + b * partner + c * (np.cos(h * partner) - 1)


def _clean(residual):
    """Classify whole bridge records, rather than individual spikes within them."""
    magnitude = np.max(np.abs(residual), axis=1) if residual.ndim == 2 else np.abs(residual)
    center = np.median(magnitude)
    mad = np.median(np.abs(magnitude - center))
    return magnitude <= center + max(0.002, 12 * mad)


def _fit(calibration):
    x = np.asarray(calibration['true_channel_loads'], dtype=float)
    y = np.asarray(calibration['bridge_raw'], dtype=float)
    sweep = calibration['single_channel_sweeps']
    s = np.asarray(sweep['true_stress'], dtype=float)
    z = np.asarray(sweep['cmd_stress'], dtype=float)
    channel = np.asarray(sweep['lab_id'], dtype=int)

    # A bounded-influence pilot separates gross spikes before estimating the
    # shared bridge shape. Sweep records constrain its uncoupled response.
    shape0 = np.array([0.25, 2.5, 0.21, 0.06, 2.0])
    u = _response(x, shape0)
    v = s + shape0[0] * np.sin(shape0[1] * s)
    offset, gain = np.empty(4), np.empty(4)
    for j in _CHANNEL:
        mask = channel == j
        t, r = np.r_[u[:, j], v[mask]], np.r_[y[:, j], z[mask]]
        offset[j], gain[j] = least_squares(
            lambda p: p[0] + p[1] * t - r, [0., 1.], loss='soft_l1', f_scale=0.002
        ).x
    rows = _clean(offset + gain * u - y)
    scans = _clean(offset[channel] + gain[channel] * v - z)
    x, y = x[rows], y[rows]
    s, z, channel = s[scans], z[scans], channel[scans]

    def residual(p):
        shift, scale, shape = p[:4], p[4:8], p[8:]
        a, k = shape[:2]
        bridge = shift + scale * _response(x, shape) - y
        single = shift[channel] + scale[channel] * (s + a * np.sin(k * s)) - z
        return np.r_[bridge.ravel(), single]

    p = least_squares(residual, np.r_[offset, gain, shape0], loss='soft_l1', f_scale=0.001).x
    return p[:4], p[4:8], p[8:]


def _invert(raw, fit):
    offset, gain, shape = fit
    raw = np.asarray(raw, dtype=float)
    target = ((raw - offset) / gain).reshape(-1, 4)
    a, k, b, c, h = shape
    linear = np.eye(4)
    linear[_CHANNEL, _PAIR] = b
    x = np.linalg.solve(linear, (target + c).T).T
    # Full Newton steps are fast near the root. The fallback takes a globally
    # contractive diagonal step when the calibrated Jacobian dominates its
    # off-diagonal term, as it does for this bridge.
    damping = 1.0 / (1 + abs(a * k))
    for _ in range(800):
        err = _response(x, shape) - target
        norm = np.max(np.abs(err), axis=1)
        active = norm > 1e-11
        if not np.any(active):
            return x.reshape(raw.shape)
        work = x[active]
        jac = np.zeros((len(work), 4, 4))
        jac[:, _CHANNEL, _CHANNEL] = 1 + a * k * np.cos(k * work)
        jac[:, _CHANNEL, _PAIR] = b - c * h * np.sin(h * work[:, _PAIR])
        step = np.linalg.solve(jac, err[active, :, None])[..., 0]
        trial = work - step
        improved = np.max(np.abs(_response(trial, shape) - target[active]), axis=1) < norm[active]
        x[active] = np.where(improved[:, None], trial, work - damping * err[active])
    raise RuntimeError('Bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return the downstream dataset with load fields in canonical units."""
    original = dataset['calibration_load_cells']
    fit = _fit(original)
    result = dataset.copy()
    calibration = original.copy()
    calibration['bridge_raw'] = np.asarray(original['true_channel_loads']).copy()
    sweep = original['single_channel_sweeps'].copy()
    sweep['cmd_stress'] = np.asarray(sweep['true_stress']).copy()
    calibration['single_channel_sweeps'] = sweep
    result['calibration_load_cells'] = calibration
    for key in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(key)
        if block is not None:
            result[key] = dict(block,
                               past_channel_raw=_invert(block['past_channel_raw'], fit),
                               hold_channel_raw=_invert(block['hold_channel_raw'], fit))
    return result
