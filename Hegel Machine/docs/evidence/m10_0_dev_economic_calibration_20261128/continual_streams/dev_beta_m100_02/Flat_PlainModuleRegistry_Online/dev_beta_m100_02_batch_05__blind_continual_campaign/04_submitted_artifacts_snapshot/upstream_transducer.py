"""Fit the paired rosette response and convert readings to reference coordinates."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

_PAIR = [2, 3, 0, 1]


def _features(x, freq):
    u = x[:, _PAIR]
    return np.stack((np.ones_like(x), x, np.sin(freq[0] * x),
                     u, 1 - np.cos(freq[1] * u)), axis=-1)


def _predict(x, freq, coef):
    return np.einsum('nck,ck->nc', _features(x, freq), coef)


def _fit(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    freq = np.array([2.5, 2.0])
    weights = np.ones(len(x))

    def profile(f, weights):
        a = _features(x, f)
        w = np.sqrt(weights)
        return np.array([np.linalg.lstsq(a[:, j] * w[:, None], y[:, j] * w,
                                        rcond=None)[0] for j in range(4)])

    # Row weights prevent an EMI spike from contaminating all four gains.
    for _ in range(20):
        coef = profile(freq, weights)
        err = np.linalg.norm(_predict(x, freq, coef) - y, axis=1)
        cutoff = max(2 * np.median(err), 1e-8)
        weights = np.minimum(1, cutoff / np.maximum(err, 1e-15))
    clean = err <= 8 * cutoff
    x, y = x[clean], y[clean]
    weights = np.ones(len(x))
    for _ in range(2):
        def residual(f):
            return ((_predict(x, f, profile(f, weights)) - y)
                    * np.sqrt(weights[:, None])).ravel()

        freq = least_squares(residual, freq, bounds=([1.8, 1.2], [3.2, 2.8]),
                             max_nfev=60).x
        coef = profile(freq, weights)
        err = np.linalg.norm(_predict(x, freq, coef) - y, axis=1)
        weights = np.minimum(1, max(2 * np.median(err), 1e-8)
                             / np.maximum(err, 1e-15))
    return freq, coef


def _invert(raw, freq, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    bias, own, sine, cross, cosine = coef.T
    x = (y - bias) / (own + freq[0] * sine)
    for _ in range(60):
        r = _predict(x, freq, coef) - y
        err = np.max(np.abs(r), axis=1)
        active = err > 2e-12
        if not np.any(active):
            break
        t, rr = x[active], r[active]
        a = own + freq[0] * sine * np.cos(freq[0] * t)
        b = cross + freq[1] * cosine * np.sin(freq[1] * t[:, _PAIR])
        # Each opposite-channel pair forms a 2-by-2 Newton system.
        step = (a[:, _PAIR] * rr - b * rr[:, _PAIR]) / (a * a[:, _PAIR] - b * b[:, _PAIR])
        scale = np.ones(len(t))
        for _ in range(18):
            cand = t - scale[:, None] * step
            bad = np.max(np.abs(_predict(cand, freq, coef) - y[active]), axis=1) >= err[active]
            if not np.any(bad):
                break
            scale[bad] *= 0.5
        x[active] = cand
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated study without mutating the input or its sweeps."""
    out = dict(study)
    sweeps = dict(study['sensor_calibration_sweeps'])
    freq, coef = _fit(sweeps['ref_rosette'], sweeps['raw_rosette'])
    out['y_obs_raw'] = _invert(study['y_obs_raw'], freq, coef)
    if study.get('reference_tracer_pulses') is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], freq, coef))
            if pulse['y_pulse_raw'] is not None else dict(pulse)
            for pulse in study['reference_tracer_pulses']
        ]
    sweeps['raw_rosette'] = np.asarray(sweeps['ref_rosette']).copy()
    banks = sweeps['single_bank_sweeps']

    def identity(bank):
        return dict(bank, raw_readout=np.asarray(bank['ref_signal']).copy())

    if isinstance(banks, dict):
        sweeps['single_bank_sweeps'] = (identity(banks) if 'ref_signal' in banks
                                        else {k: identity(v) for k, v in banks.items()})
    else:
        sweeps['single_bank_sweeps'] = type(banks)(identity(bank) for bank in banks)
    out['sensor_calibration_sweeps'] = sweeps
    return out
