"""Robust four-channel rosette calibration before transport inversion."""

import numpy as np


_PAIRS = np.triu_indices(4)


def _design(x):
    i, j = _PAIRS
    return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j], x**3))


def _upper_fence(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median + 8 * max(1.4826 * mad, 1e-10)


def _fit(ref, raw):
    design = _design(ref)
    # Reject gross EMI before fitting, then screen for less obvious spikes
    # by their joint residual across the four channels.
    norms = np.linalg.norm(raw - np.median(raw, axis=0), axis=1)
    keep = norms <= _upper_fence(norms)
    coef = np.linalg.lstsq(design[keep], raw[keep], rcond=None)[0]
    residual = np.linalg.norm(design @ coef - raw, axis=1)
    keep = residual <= _upper_fence(residual[keep])
    return np.linalg.lstsq(design[keep], raw[keep], rcond=None)[0]


def _jacobian(x, coef):
    jac = np.broadcast_to(coef[1:5].T, (len(x), 4, 4)).copy()
    for k, (i, j) in enumerate(zip(*_PAIRS), start=5):
        jac[:, :, i] += x[:, j, None] * coef[k]
        jac[:, :, j] += x[:, i, None] * coef[k]
    jac += np.eye(4) * (3 * x**2)[:, None, :] * coef[-4:].T
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T.copy()
    for _ in range(35):
        residual = _design(x) @ coef - y
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-11 * (1 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            return x.reshape(raw.shape)
        idx = np.flatnonzero(active)
        step = np.linalg.solve(_jacobian(x[idx], coef), residual[idx, :, None])[..., 0]
        for _ in range(20):
            trial = x[idx] - step
            better = np.linalg.norm(_design(trial) @ coef - y[idx], axis=1) < error[idx]
            x[idx[better]] = trial[better]
            idx, step = idx[~better], 0.5 * step[~better]
            if not len(idx):
                break
    raise ValueError('Rosette inverse did not converge')


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    banks = sweeps['single_bank_sweeps']
    result = dict(study)
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(banks, raw_readout=np.asarray(banks['ref_signal']).copy()),
    )
    result['y_obs_raw'] = _invert(study['y_obs_raw'], coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef)) for pulse in pulses
        ]
    return result
