"""Calibrate the four-channel rosette into the PDE solver's coordinates."""

import numpy as np
from scipy.optimize import least_squares


_PAIR = np.arange(4) ^ 2


def _basis(x, z, f):
    return np.stack((np.ones_like(x), x, np.sin(f[0] * x),
                     z, -2 * np.sin(0.5 * f[1] * z)**2), axis=-1)


def _gross_good(y):
    deviation = np.abs(y - np.median(y, axis=0))
    mad = np.median(deviation, axis=0)
    return deviation <= 8 * np.maximum(mad, np.median(mad) * 0.5)


def _fit(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    good = np.all(_gross_good(y), axis=1)
    bank = sweeps['single_bank_sweeps']
    u, v, ids = (np.asarray(bank[k]) for k in ('ref_signal', 'raw_readout', 'bank_id'))
    bank_good = np.zeros(len(v), dtype=bool)
    for j in range(4):
        here = ids == j
        if here.any():
            bank_good[here] = _gross_good(v[here])

    data = []
    for j, pair in enumerate(_PAIR):
        use = bank_good & (ids == j)
        data.append((np.r_[x[good, j], u[use]],
                     np.r_[x[good, pair], np.zeros(use.sum())],
                     np.r_[y[good, j], v[use]]))
    weights = [np.ones_like(t) for _, _, t in data]

    def profile(f, coefficients=False):
        parts = []
        for (a, b, t), w in zip(data, weights):
            d = _basis(a, b, f)
            coef = np.linalg.lstsq(d * w[:, None], t * w, rcond=None)[0]
            parts.append(coef if coefficients else w * (d @ coef - t))
        return np.stack(parts) if coefficients else np.concatenate(parts)

    # Profile out the linear gains and offsets. Downweight any smaller EMI
    # spikes that survived the initial gross-row screen.
    f = np.array([2.5, 2.0])
    for step in range(5):
        f = least_squares(profile, f, bounds=(0.7, 4.0),
                          ftol=1e-11, gtol=1e-12).x
        coef = profile(f, coefficients=True)
        if step == 4:
            break
        errors = [_basis(a, b, f) @ c - t for (a, b, t), c in zip(data, coef)]
        n = good.sum()
        rows = np.max(np.abs(np.column_stack([e[:n] for e in errors])), axis=1)
        err = np.concatenate(errors)
        scale = max(1e-10, 1.4826 * np.median(np.abs(err - np.median(err))))
        weights = [1 / np.hypot(1, np.r_[rows, e[n:]] / (6 * scale)) for e in errors]
    return f, coef


def _invert(raw, f, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    offset, gain, sin_gain, cross, cos_gain = coef.T
    x = (y - offset) / (gain + f[0] * sin_gain)

    def residual(t):
        z = t[:, _PAIR]
        return (offset + gain * t + sin_gain * np.sin(f[0] * t) + cross * z
                - 2 * cos_gain * np.sin(0.5 * f[1] * z)**2 - y)

    for _ in range(48):
        r = residual(x)
        if not np.any(np.abs(r) > 1e-12):
            break
        d = gain + f[0] * sin_gain * np.cos(f[0] * x)
        c = cross - f[1] * cos_gain * np.sin(f[1] * x[:, _PAIR])
        direction = (d[:, _PAIR] * r - c * r[:, _PAIR]) / (d * d[:, _PAIR] - c * c[:, _PAIR])
        # The rosette has two independent coupled pairs. Backtrack each pair
        # separately so a mildly saturated pair cannot stall the other.
        cost = r*r + r[:, _PAIR]**2
        trial = x - direction
        for _ in range(20):
            q = residual(trial)**2
            worse = (q + q[:, _PAIR] > cost) & (cost > 1e-24)
            if not np.any(worse):
                break
            direction[worse] *= 0.5
            trial[worse] = x[worse] - direction[worse]
        x = trial
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated study without modifying the supplied measurements."""
    sweeps = study['sensor_calibration_sweeps']
    f, coef = _fit(sweeps)
    out = dict(study)
    out['y_obs_raw'] = _invert(study['y_obs_raw'], f, coef)
    bank = sweeps['single_bank_sweeps']
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(bank, raw_readout=bank['ref_signal'].copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], f, coef))
            if pulse is not None and pulse['y_pulse_raw'] is not None else pulse
            for pulse in pulses]
    return out
