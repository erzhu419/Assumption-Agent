"""Robust four-channel calibration for the chamber rosette."""

import numpy as np


_I, _J = np.triu_indices(4)
_EYE = np.eye(4)


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit(ref, raw):
    a = _basis(ref)
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Weight whole rows: the EMI bursts affect multiple channels together.
    for _ in range(25):
        error = np.linalg.norm(a @ coef - raw, axis=1)
        scale = max(1.8 * np.median(error), 1e-12)
        weight = 1 / np.hypot(1, error / scale)
        coef = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
    error = np.linalg.norm(a @ coef - raw, axis=1)
    clean = error <= max(8 * np.median(error), 1e-12)
    if np.linalg.matrix_rank(a[clean]) == a.shape[1]:
        coef = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
    return coef


def _jacobian(x, coef):
    q_derivative = (x[:, _J, None] * _EYE[_I] + x[:, _I, None] * _EYE[_J])
    q = np.einsum('nki,kj->nji', q_derivative, coef[5:15])
    return coef[1:5].T + q + 3 * x[:, None, :]**2 * coef[15:].T


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    if not len(y):
        return raw.copy()
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    residual = _basis(x) @ coef - y
    for _ in range(30):
        error = np.linalg.norm(residual, axis=1)
        active = error > 2e-13 * (1 + np.linalg.norm(y, axis=1))
        if not active.any():
            break
        xa, ya = x[active], y[active]
        delta = np.linalg.solve(_jacobian(xa, coef), residual[active, :, None])[..., 0]
        step = np.ones((len(xa), 1))
        for _ in range(16):
            trial = xa - step * delta
            change = _basis(trial) @ coef - ya
            worse = np.linalg.norm(change, axis=1) > error[active]
            if not worse.any():
                break
            step[worse] *= 0.5
        x[active] = np.where(worse[:, None], xa, trial)
        residual[active] = np.where(worse[:, None], residual[active], change)
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a study in reference coordinates without changing the input."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    coef = _fit(ref, np.asarray(sweeps['raw_rosette']))
    out = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coef))
    normalized = dict(sweeps, raw_rosette=ref.copy())
    bank = sweeps['single_bank_sweeps']
    normalized['single_bank_sweeps'] = dict(bank, raw_readout=bank['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = normalized
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
                                          for pulse in pulses]
    return out
