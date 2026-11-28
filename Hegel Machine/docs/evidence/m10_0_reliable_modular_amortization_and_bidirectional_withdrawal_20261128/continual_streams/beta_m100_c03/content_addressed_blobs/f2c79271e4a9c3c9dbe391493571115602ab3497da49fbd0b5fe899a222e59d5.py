"""Fit the study's rosette response and put raw measurements in reference units."""

import numpy as np

_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4
_PREV = (_CHANNEL - 1) % 4
_I, _J = np.triu_indices(4)


def _design(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _response(a, y, clean=None):
    # Weight entire rows together: EMI affects multiple outputs at once.
    if clean is None:
        weights = np.ones(len(a))
        for _ in range(12):
            coef = np.linalg.lstsq(a * weights[:, None], y * weights[:, None], rcond=None)[0]
            err = np.linalg.norm(a @ coef - y, axis=1)
            scale = max(2.0 * np.median(err), 1e-10)
            weights = 1.0 / np.hypot(1.0, err / scale)
        clean = err <= 4.0 * scale
    return np.linalg.lstsq(a[clean], y[clean], rcond=None)[0], clean


def _fit(x, y):
    a = _design(x)
    full, clean = _response(a, y)
    cyclic = np.zeros_like(full)
    for j, k, p in zip(_CHANNEL, _NEXT, _PREV):
        pair = 5 + np.flatnonzero((_I == min(j, k)) & (_J == max(j, k)))[0]
        terms = [0, 1+j, 1+k, 1+p, pair, 15+j]
        cyclic[terms, j] = np.linalg.lstsq(a[clean][:, terms], y[clean, j], rcond=None)[0]
    # Prefer the observed ring structure only when it explains the clean sweeps
    # to the same accuracy as the broader polynomial model.
    noise = np.linalg.norm(a[clean] @ full - y[clean])
    if np.linalg.norm(a[clean] @ cyclic - y[clean]) <= 1.3 * noise + 1e-8:
        return cyclic
    return full


def _jacobian(x, coef):
    jac = np.broadcast_to(coef[1:5].T, (len(x), 4, 4)).copy()
    for q, (i, j) in enumerate(zip(_I, _J), 5):
        jac[:, :, i] += x[:, j, None] * coef[q]
        jac[:, :, j] += x[:, i, None] * coef[q]
    jac += 3.0 * coef[15:].T * x[:, None, :]**2
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    for _ in range(30):
        err = _design(x) @ coef - y
        norm = np.linalg.norm(err, axis=1)
        active = norm > 1e-12 * (1.0 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            break
        z, target = x[active], y[active]
        step = np.linalg.solve(_jacobian(z, coef), err[active, :, None])[..., 0]
        old = norm[active]
        rate = np.ones(len(z))
        for _ in range(16):
            candidate = z - rate[:, None] * step
            improve = np.linalg.norm(_design(candidate) @ coef - target, axis=1) < old
            if np.all(improve):
                break
            rate[~improve] *= 0.5
        x[active] = np.where(improve[:, None], candidate, z)
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Convert observations and tracer pulses without mutating the input study."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    result = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coef))
    calibrated_sweeps = dict(sweeps, raw_rosette=sweeps['ref_rosette'].copy())
    banks = sweeps.get('single_bank_sweeps')
    if banks is not None:
        calibrated_sweeps['single_bank_sweeps'] = dict(banks, raw_readout=banks['ref_signal'].copy())
    result['sensor_calibration_sweeps'] = calibrated_sweeps
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return result
