"""Calibrate a four-channel rosette before the canonical PDE solve."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


_OPPOSITE = [2, 3, 0, 1]


def _forward(x, shape):
    a, k, c, d, h = shape
    z = x[..., _OPPOSITE]
    return x + a * np.sin(k * x) + c * z + d * (1 - np.cos(h * z))


def _fit(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    single = sweeps['single_bank_sweeps']
    bank = np.asarray(single['bank_id'], dtype=int).ravel()
    u = np.asarray(single['ref_signal'], dtype=float).ravel()
    v = np.asarray(single['raw_readout'], dtype=float).ravel()

    def residual(p):
        bias, gain = p[:4], p[4:8]
        a, k = p[8:10]
        return (bias + gain * _forward(x, p[8:]) - y,
                bias[bank] + gain[bank] * (u + a * np.sin(k * u)) - v)

    p0 = np.r_[np.zeros(4), np.ones(4), .25, 2.5, -.21, .06, 2.]
    # These bounds keep the normalized response strictly diagonally dominant:
    # the maximum summed nonlinear and cross-channel slope is below one.
    lower = np.r_[np.full(4, -np.inf), np.full(4, 1e-6), .18, 2.2, -.24, .03, 1.7]
    upper = np.r_[np.full(8, np.inf), .27, 2.6, -.17, .07, 2.2]
    bounds = (lower, upper)

    def all_residuals(p):
        r, t = residual(p)
        return np.r_[r.ravel(), t]

    p = least_squares(all_residuals, p0, bounds=bounds,
                      loss='soft_l1', f_scale=.003).x
    r, t = residual(p)
    # Reject whole rosette rows: EMI can corrupt more than one channel.
    dev = np.abs(np.r_[r.ravel(), t])
    cutoff = max(10 * np.median(dev), 1e-4)
    keep_r = np.max(np.abs(r), axis=1) <= cutoff
    keep_t = np.abs(t) <= cutoff

    def clean_residuals(p):
        r, t = residual(p)
        return np.r_[r[keep_r].ravel(), t[keep_t]]

    p = least_squares(clean_residuals, p, bounds=bounds).x
    return p[:4], p[4:8], p[8:]


def _inverse(raw, bias, gain, shape):
    raw = np.asarray(raw, dtype=float)
    y = (raw.reshape(-1, 4) - bias) / gain
    x = y.copy()
    a, k, c, d, h = shape
    # Solve the two opposite-bank pairs in parallel. Limit the Newton step
    # if necessary; a residual step is contractive under the fit bounds.
    for _ in range(60):
        r = _forward(x, shape) - y
        norm = np.max(np.abs(r), axis=1, keepdims=True)
        if np.max(norm, initial=0.) <= 2e-13:
            return x.reshape(raw.shape)
        diag = 1 + a * k * np.cos(k * x)
        off = c + d * h * np.sin(h * x[:, _OPPOSITE])
        det = diag * diag[:, _OPPOSITE] - off * off[:, _OPPOSITE]
        step = (diag[:, _OPPOSITE] * r - off * r[:, _OPPOSITE]) / det
        fixed = x - r
        newton = x - step
        fixed_norm = np.max(np.abs(_forward(fixed, shape) - y), axis=1, keepdims=True)
        newton_norm = np.max(np.abs(_forward(newton, shape) - y), axis=1, keepdims=True)
        x = np.where(newton_norm <= fixed_norm, newton, fixed)
    # Finish by contraction if a row is far outside the sweep range.
    for _ in range(4000):
        r = _forward(x, shape) - y
        if np.max(np.abs(r), initial=0.) <= 2e-13:
            break
        x -= r
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a canonical-coordinate study without modifying the input."""
    sweeps = study['sensor_calibration_sweeps']
    params = _fit(sweeps)
    out = dict(study)
    out['y_obs_raw'] = _inverse(study['y_obs_raw'], *params)
    single = sweeps['single_bank_sweeps']
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=np.asarray(sweeps['ref_rosette']).copy(),
        single_bank_sweeps=dict(single, raw_readout=np.asarray(single['ref_signal']).copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], *params))
            if pulse.get('y_pulse_raw') is not None else dict(pulse)
            for pulse in pulses
        ]
    return out
