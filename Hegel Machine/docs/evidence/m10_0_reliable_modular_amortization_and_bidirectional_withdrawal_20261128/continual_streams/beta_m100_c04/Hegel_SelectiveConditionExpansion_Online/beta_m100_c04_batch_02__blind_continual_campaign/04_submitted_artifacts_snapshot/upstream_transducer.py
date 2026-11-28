"""Fit and invert the coupled four-channel rosette response."""

import numpy as np

_LEFT, _RIGHT = np.triu_indices(4, k=1)


def _design(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _LEFT] * x[:, _RIGHT], x**3))


def _fit(ref, raw):
    design = _design(ref)
    coeff = np.linalg.lstsq(design, raw, rcond=None)[0]
    # Downweight whole rows: EMI perturbs multiple outputs in the same sweep.
    for _ in range(24):
        residual = np.linalg.norm(design @ coeff - raw, axis=1)
        cutoff = max(1.5 * np.median(residual), 1e-12)
        weight = np.sqrt(cutoff / np.maximum(residual, cutoff))[:, None]
        coeff = np.linalg.lstsq(design * weight, raw * weight, rcond=None)[0]
    residual = np.linalg.norm(design @ coeff - raw, axis=1)
    clean = residual <= max(6 * np.median(residual), 1e-11)
    return np.linalg.lstsq(design[clean], raw[clean], rcond=None)[0]


def _jacobian(x, coeff):
    # Rows index input channels, columns index output channels.
    jac = coeff[1:5][None, :, :] + 3 * x[:, :, None]**2 * coeff[11:][None, :, :]
    for k, (i, j) in enumerate(zip(_LEFT, _RIGHT)):
        jac[:, i, :] += x[:, j, None] * coeff[5 + k]
        jac[:, j, :] += x[:, i, None] * coeff[5 + k]
    return jac.transpose(0, 2, 1)


def _invert(raw, coeff):
    original = np.asarray(raw, dtype=float)
    target = original.reshape(-1, 4)
    x = np.linalg.solve(coeff[1:5].T, (target - coeff[0]).T).T
    tolerance = 1e-12 * (1 + np.max(np.abs(target), axis=1))
    for _ in range(40):
        residual = _design(x) @ coeff - target
        active = np.flatnonzero(np.max(np.abs(residual), axis=1) > tolerance)
        if not len(active):
            return x.reshape(original.shape)
        point, rhs = x[active], residual[active]
        step = np.linalg.solve(_jacobian(point, coeff), rhs[..., None])[..., 0]
        norm = np.linalg.norm(rhs, axis=1)
        trial = point - step
        for _ in range(16):
            rejected = np.linalg.norm(_design(trial) @ coeff - target[active], axis=1) > norm
            if not np.any(rejected):
                break
            step[rejected] *= 0.5
            trial[rejected] = point[rejected] - step[rejected]
        x[active] = trial
    raise RuntimeError('rosette calibration inverse did not converge')


def calibrate_upstream(study: dict) -> dict:
    """Return the study in reference coordinates, without mutating the input."""
    cal = study['sensor_calibration_sweeps']
    ref = np.asarray(cal['ref_rosette'], dtype=float)
    coeff = _fit(ref, np.asarray(cal['raw_rosette'], dtype=float))
    banks = cal['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coeff))
    calibrated['sensor_calibration_sweeps'] = dict(
        cal, raw_rosette=cal['ref_rosette'].copy(),
        single_bank_sweeps=dict(banks, raw_readout=banks['ref_signal'].copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coeff))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return calibrated
