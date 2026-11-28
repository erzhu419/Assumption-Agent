"""Robust calibration of the four-channel rosette into reference coordinates."""

import numpy as np
from scipy.optimize import least_squares


_PAIR = np.array([2, 3, 0, 1])


def _design(x, freq):
    z = x[:, _PAIR]
    return np.stack((np.ones_like(x), x, np.sin(freq[0] * x),
                     z, 1 - np.cos(freq[1] * z)), axis=-1).transpose(1, 0, 2)


def _fit(x, y):
    """Profile channel gains and offsets, sharing the two shape frequencies."""
    freq = np.array([2.5, 2.0])
    weight = np.ones(len(x))
    for _ in range(10):
        root = np.sqrt(weight)

        def profile(f):
            a = _design(x, f)
            coef = np.stack([np.linalg.lstsq(ai * root[:, None], yi * root,
                                             rcond=None)[0]
                             for ai, yi in zip(a, y.T)])
            residual = np.einsum('cnp,cp->nc', a, coef) - y
            return residual, coef

        freq = least_squares(lambda f: (profile(f)[0] * root[:, None]).ravel(),
                             freq, bounds=([1.8, 1.2], [3.3, 3.0]),
                             ftol=1e-10, gtol=1e-10).x
        residual, coef = profile(freq)
        size = np.linalg.norm(residual, axis=1)
        scale = max(3 * np.median(size), 1e-10)
        new_weight = 1 / (1 + (size / scale)**2)
        if np.max(abs(new_weight - weight)) < 1e-4:
            break
        weight = new_weight
    return freq, coef


def _inverse(raw, freq, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    offset, gain, amp, cross, bend = coef.T
    k, h = freq
    x = (y - offset) / gain
    # On each pair, solve both channels jointly. The measured law is strictly
    # diagonally dominant: the smallest self slope exceeds the largest cross slope.
    for _ in range(50):
        z = x[:, _PAIR]
        r = offset + gain * x + amp * np.sin(k * x) + cross * z + bend * (1 - np.cos(h * z)) - y
        a = gain + k * amp * np.cos(k * x)
        b = cross + h * bend * np.sin(h * z)
        step = (a[:, _PAIR] * r - b * r[:, _PAIR]) / (a * a[:, _PAIR] - b * b[:, _PAIR])
        if np.max(abs(step), initial=0) < 1e-12:
            break
        # Backtrack per row so a near-saturation row cannot slow the rest.
        old = np.sum(r * r, axis=1)
        trial = x - step
        for _ in range(16):
            t = trial[:, _PAIR]
            err = offset + gain * trial + amp * np.sin(k * trial) + cross * t + bend * (1 - np.cos(h * t)) - y
            worse = np.sum(err * err, axis=1) > old
            if not np.any(worse):
                break
            trial[worse] = (trial[worse] + x[worse]) / 2
        x = trial
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated study without modifying the input measurements."""
    sweeps = study['sensor_calibration_sweeps']
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    freq, coef = _fit(x, y)
    out = dict(study)
    out['y_obs_raw'] = _inverse(study['y_obs_raw'], freq, coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(p, y_pulse_raw=_inverse(p['y_pulse_raw'], freq, coef))
            if p.get('y_pulse_raw') is not None else dict(p) for p in pulses
        ]
    bank = sweeps['single_bank_sweeps']
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=x.copy(), single_bank_sweeps=dict(
            bank, raw_readout=np.asarray(bank['ref_signal']).copy()))
    return out
