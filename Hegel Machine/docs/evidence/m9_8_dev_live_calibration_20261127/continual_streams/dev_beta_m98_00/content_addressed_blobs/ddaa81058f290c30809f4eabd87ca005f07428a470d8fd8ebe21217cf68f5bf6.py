"""Robust forward calibration and inversion of the coupled four-channel rosette."""

import numpy as np


_I, _J = np.triu_indices(4)


def _features(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit(ref, raw):
    a = _features(ref)
    scale = np.maximum(np.sqrt(np.mean(a*a, axis=0)), 1e-12)
    a = a / scale
    weight = np.ones(len(a))
    for _ in range(25):
        coef = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(a @ coef - raw, axis=1)
        noise = max(np.median(error), 1e-10)
        weight = 1 / np.hypot(1, error / (2 * noise))

    # Reject EMI by row, then remove only nonlinear terms below the noise scale.
    clean = error <= 6 * noise
    if np.count_nonzero(clean) >= a.shape[1]:
        coef = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
        for j in range(4):
            keep = np.abs(coef[:, j]) > 4 * noise
            keep[:5] = True
            coef[~keep, j] = 0
            coef[keep, j] = np.linalg.lstsq(a[clean][:, keep], raw[clean, j], rcond=None)[0]
    return coef / scale[:, None]


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    linear = coef[1:5].T
    x = np.linalg.solve(linear, (target - coef[0]).T).T.copy()
    quad, cubic = coef[5:15], coef[15:19]
    for _ in range(24):
        residual = _features(x) @ coef - target
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-12 * (1 + np.linalg.norm(target, axis=1))
        if not np.any(active):
            break
        xa, ra = x[active], residual[active]
        jac = np.broadcast_to(linear, (len(xa), 4, 4)).copy()
        for i, j, q in zip(_I, _J, quad):
            jac[:, :, i] += xa[:, j, None] * q
            jac[:, :, j] += xa[:, i, None] * q
        jac += cubic.T[None, :, :] * (3 * xa[:, None, :]**2)
        step = np.linalg.solve(jac, ra[..., None])[..., 0]
        alpha = np.ones((len(xa), 1))
        for _ in range(14):
            trial = xa - alpha * step
            worse = np.linalg.norm(_features(trial) @ coef - target[active], axis=1) >= error[active]
            if not np.any(worse):
                break
            alpha[worse] *= 0.5
        trial[worse] = xa[worse]
        x[active] = trial
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return an identity-calibrated study without modifying the input."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    out = dict(study)
    out['y_obs_raw'] = _invert(study['y_obs_raw'], coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse['y_pulse_raw'] is not None else dict(pulse)
            for pulse in pulses
        ]
    canonical = dict(sweeps, raw_rosette=sweeps['ref_rosette'].copy())
    if 'single_bank_sweeps' in sweeps:
        bank = sweeps['single_bank_sweeps']
        canonical['single_bank_sweeps'] = dict(bank, raw_readout=bank['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = canonical
    return out
