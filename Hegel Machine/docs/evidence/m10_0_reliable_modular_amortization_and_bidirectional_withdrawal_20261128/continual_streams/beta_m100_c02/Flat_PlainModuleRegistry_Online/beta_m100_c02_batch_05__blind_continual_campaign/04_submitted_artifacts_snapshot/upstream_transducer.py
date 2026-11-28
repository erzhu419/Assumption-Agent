"""Fit and invert a four-bank rosette with opposite-bank crosstalk."""
import numpy as np
from scipy.optimize import least_squares


def _clean(residual):
    center = np.median(residual)
    mad = np.median(np.abs(residual - center))
    return residual <= center + 8 * max(mad, 1e-8)


def _response(x, shape):
    a, b = shape
    return x + a * np.sin(b*x)


def _design(x, shape, freq):
    q = np.roll(x, 2, axis=-1)
    return np.stack((np.ones_like(x), _response(x, shape), q,
                     1 - np.cos(freq*q)), axis=-1)


def _fit(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    single = sweeps['single_bank_sweeps']
    u = np.asarray(single['ref_signal']).ravel()
    v = np.asarray(single['raw_readout']).ravel()
    banks = np.asarray(single['bank_id'], dtype=int).ravel()

    def scalar_resid(p):
        return p[2:6][banks] + p[6:10][banks] * _response(u, p[:2]) - v

    p = least_squares(scalar_resid, np.r_[.25, 2.5, np.zeros(4), np.ones(4)],
                      loss='soft_l1', f_scale=.01).x
    keep = _clean(np.abs(scalar_resid(p)))
    p = least_squares(lambda p: scalar_resid(p)[keep], p).x
    shape = p[:2]

    def rosette_resid(params):
        return np.sum(_design(x, shape, params[0]) * params[1:].reshape(4, 4), axis=-1) - y

    start = np.column_stack((p[2:6], p[6:10], -.2*p[6:10], .06*p[6:10]))
    params = least_squares(lambda t: rosette_resid(t).ravel(), np.r_[2., start.ravel()],
                           loss='soft_l1', f_scale=.01).x
    rows = _clean(np.max(np.abs(rosette_resid(params)), axis=1))
    params = least_squares(lambda t: rosette_resid(t)[rows].ravel(), params,
                           gtol=1e-12).x
    return shape, params[0], params[1:].reshape(4, 4)


def _forward(x, shape, freq, coef):
    return np.sum(_design(x, shape, freq) * coef, axis=-1)


def _inverse(raw, shape, freq, coef):
    raw = np.asarray(raw, dtype=float)
    values = raw.reshape(-1, 4)
    ans = np.empty_like(values)
    bias, gain, linear, even = coef.T
    j = np.arange(4)
    opp = (j + 2) % 4
    a, b = shape

    def jac(x):
        derivative = np.diag(gain * (1 + a*b*np.cos(b*x)))
        derivative[j, opp] = linear + even*freq*np.sin(freq*x[opp])
        return derivative

    for i, target in enumerate(values):
        guess = (target - bias) / gain
        ans[i] = least_squares(lambda x: _forward(x, shape, freq, coef) - target,
                               guess, jac=jac, gtol=1e-12, ftol=1e-12, xtol=1e-12).x
    return ans.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a canonical-coordinate study without changing the input."""
    sweeps = study['sensor_calibration_sweeps']
    model = _fit(sweeps)
    out = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], *model))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], *model))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    single = sweeps['single_bank_sweeps']
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(single, raw_readout=single['ref_signal'].copy()),
    )
    return out
