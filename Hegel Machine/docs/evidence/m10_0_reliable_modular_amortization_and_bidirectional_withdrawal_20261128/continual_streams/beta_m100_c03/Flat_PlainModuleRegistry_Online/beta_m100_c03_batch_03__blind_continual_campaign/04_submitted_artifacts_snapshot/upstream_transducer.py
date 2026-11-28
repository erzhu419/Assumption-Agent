"""Robust four-channel rosette calibration in reference coordinates."""

import numpy as np


_I, _J = np.tril_indices(4, -1)


def _features(x):
    return np.column_stack((np.ones(len(x)), x, x**3, x[:, _I] * x[:, _J]))


def _fit(ref, raw):
    design = _features(ref)
    weight = np.ones(len(ref))
    # Score complete rows: EMI in one channel can contaminate the whole sweep.
    for _ in range(20):
        coef = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(raw - design @ coef, axis=1)
        scale = max(2 * np.median(error), 1e-12)
        weight = 1 / (1 + (error / scale)**2)
    return coef


def _jacobian(x, coef):
    jac = coef[1:5].T[None, :, :] + 3 * coef[5:9].T[None, :, :] * x[:, None, :]**2
    for i, j, term in zip(_I, _J, coef[9:]):
        jac[:, :, i] += x[:, j, None] * term
        jac[:, :, j] += x[:, i, None] * term
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    for _ in range(35):
        residual = _features(x) @ coef - y
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-11 * (1 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            break
        xa, ya, ea = x[active], y[active], error[active]
        step = np.linalg.solve(_jacobian(xa, coef), residual[active, :, None])[..., 0]
        proposal = xa - step
        for _ in range(16):
            bad = np.linalg.norm(_features(proposal) @ coef - ya, axis=1) >= ea
            if not np.any(bad):
                break
            step[bad] *= 0.5
            proposal[bad] = xa[bad] - step[bad]
        accepted = np.linalg.norm(_features(proposal) @ coef - ya, axis=1) < ea
        if not np.any(accepted):
            break
        xa[accepted] = proposal[accepted]
        x[active] = xa
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Invert observations and tracer pulses without modifying the input study."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    result = dict(study)
    result['y_obs_raw'] = _invert(study['y_obs_raw'], coef)
    bank = sweeps['single_bank_sweeps']
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(bank, raw_readout=np.asarray(bank['ref_signal']).copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return result
