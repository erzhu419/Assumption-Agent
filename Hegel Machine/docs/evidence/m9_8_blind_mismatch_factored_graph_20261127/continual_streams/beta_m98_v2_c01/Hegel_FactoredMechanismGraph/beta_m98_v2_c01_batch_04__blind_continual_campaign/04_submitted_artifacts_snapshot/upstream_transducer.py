"""Fit and invert the coupled rosette response before tomography."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


_QUADRATIC = tuple(combinations_with_replacement(range(4), 2))


def _design(x):
    return np.column_stack((np.ones(len(x)), x,
                            *(x[:, i] * x[:, j] for i, j in _QUADRATIC), x**3))


def _fit(ref, raw):
    design = _design(ref)
    weight = np.ones(len(ref))
    for _ in range(30):
        coef = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(raw - design @ coef, axis=1)
        scale = max(1.5 * np.median(error), 1e-8)
        weight = 1 / np.hypot(1, error / scale)

    # EMI affects whole sweep rows; refit only the consensus set.
    good = error <= 8 * scale
    a, y = design[good], raw[good]
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    noise = np.sqrt(np.mean((y - a @ coef)**2, axis=0))
    term_size = np.sqrt(np.mean(a**2, axis=0))
    for j in range(4):
        keep = np.abs(coef[:, j]) * term_size > 4 * noise[j]
        keep[:5] = True
        compact = np.linalg.lstsq(a[:, keep], y[:, j], rcond=None)[0]
        if np.linalg.norm(y[:, j] - a[:, keep] @ compact) <= 1.15 * noise[j] * np.sqrt(len(a)):
            coef[:, j] = 0
            coef[keep, j] = compact
    return coef


def _jacobian(x, coef):
    jac = coef[1:5].T.copy()
    for k, (i, j) in enumerate(_QUADRATIC, 5):
        jac[:, i] += x[j] * coef[k]
        jac[:, j] += x[i] * coef[k]
    jac += coef[-4:].T * (3 * x**2)
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (target - coef[0]).T).T
    for i, y in enumerate(target):
        def residual(z):
            return _design(z[None, :])[0] @ coef - y

        x[i] = least_squares(residual, x[i], jac=lambda z: _jacobian(z, coef),
                             gtol=1e-12, ftol=1e-12, xtol=1e-12).x
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a study in reference coordinates without mutating the input."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    single = sweeps['single_bank_sweeps']
    normalized_single = dict(single, raw_readout=np.asarray(single['ref_signal']).copy())
    normalized_sweeps = dict(sweeps, raw_rosette=ref.copy(), single_bank_sweeps=normalized_single)
    calibrated = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coef),
                      sensor_calibration_sweeps=normalized_sweeps)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef)) for pulse in pulses
        ]
    return calibrated
