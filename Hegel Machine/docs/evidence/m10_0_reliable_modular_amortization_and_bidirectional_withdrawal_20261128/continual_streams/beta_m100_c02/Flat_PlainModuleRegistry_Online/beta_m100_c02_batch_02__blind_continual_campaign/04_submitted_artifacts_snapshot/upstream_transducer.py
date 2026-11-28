"""Calibrate the coupled four-channel rosette before chamber inversion."""
from itertools import combinations

import numpy as np


_PAIRS = tuple(combinations(range(4), 2))


def _features(x):
    return np.column_stack((np.ones(len(x)), x, x**3,
                            *(x[:, i] * x[:, j] for i, j in _PAIRS)))


def _fit_forward(ref, raw):
    """Fit offset, linear mixing, self-cubic response and pair interactions."""
    a = _features(ref)
    weight = np.ones(len(ref))
    for _ in range(24):
        coef = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(a @ coef - raw, axis=1)
        scale = max(np.median(error), 1e-12)
        # Cauchy IRLS, using one weight for all channels of each sweep row.
        weight = 1 / np.hypot(1, error / (2 * scale))
    clean = error <= 6 * scale
    if np.linalg.matrix_rank(a[clean]) == a.shape[1]:
        coef = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
    return coef


def _jacobian(x, coef):
    # First derivative axis indexes inputs; second indexes outputs.
    jac = coef[1:5].copy() + 3 * x[:, None]**2 * coef[5:9]
    for term, (i, j) in zip(coef[9:], _PAIRS):
        jac[i] += x[j] * term
        jac[j] += x[i] * term
    return jac.T


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    targets = raw.reshape(-1, 4)
    result = np.linalg.solve(coef[1:5].T, (targets - coef[0]).T).T
    for x, target in zip(result, targets):
        for _ in range(30):
            residual = _features(x[None, :])[0] @ coef - target
            error = np.linalg.norm(residual)
            if error <= 1e-12 * (1 + np.linalg.norm(target)):
                break
            step = np.linalg.solve(_jacobian(x, coef), residual)
            for power in range(16):
                trial = x - 2.0**(-power) * step
                if np.linalg.norm(_features(trial[None, :])[0] @ coef - target) < error:
                    x[:] = trial
                    break
            else:
                raise RuntimeError('rosette inversion did not decrease the forward residual')
        else:
            raise RuntimeError('rosette inversion did not converge')
    return result.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return canonical observations and identity sweeps without changing the input."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit_forward(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    single = sweeps['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coef))
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(single, raw_readout=single['ref_signal'].copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse['y_pulse_raw'] is not None else pulse for pulse in pulses
        ]
    return calibrated
