"""Robust forward calibration and inversion of the four-channel rosette."""
import numpy as np


_I, _J = np.triu_indices(4, 1)


def _design(x):
    x2 = x * x
    return np.column_stack((np.ones(len(x)), x, x2, x[:, _I] * x[:, _J], x2 * x))


def _fit_forward(ref, raw):
    a = _design(ref)
    weights = np.ones(len(a))
    # Reweight whole rows: an EMI burst can affect several banks at once.
    for _ in range(24):
        coeff = np.linalg.lstsq(a * weights[:, None], raw * weights[:, None], rcond=None)[0]
        error = np.linalg.norm(raw - a @ coeff, axis=1)
        scale = max(float(np.median(error)), 1e-12)
        weights = np.minimum(1.0, 2.0 * scale / np.maximum(error, 1e-15))
    clean = error <= 5.0 * scale
    return np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]


def _jacobian(x, coeff):
    jac = (coeff[None, 1:5, :] + 2.0 * x[:, :, None] * coeff[None, 5:9, :]
           + 3.0 * x[:, :, None] ** 2 * coeff[None, 15:19, :])
    for k, (i, j) in enumerate(zip(_I, _J), start=9):
        jac[:, i, :] += x[:, j, None] * coeff[k]
        jac[:, j, :] += x[:, i, None] * coeff[k]
    return jac.swapaxes(1, 2)


def _invert(values, coeff):
    raw = np.asarray(values, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(coeff[1:5].T, (y - coeff[0]).T).T
    for _ in range(32):
        residual = _design(x) @ coeff - y
        error = np.max(np.abs(residual), axis=1)
        active = error > 1e-12
        if not np.any(active):
            break
        xa, ya, ea = x[active], y[active], error[active]
        step = np.linalg.solve(_jacobian(xa, coeff), residual[active, :, None])[..., 0]
        accepted = xa.copy()
        pending = np.arange(len(xa))
        for backtrack in range(14):
            trial = xa[pending] - np.exp2(-backtrack) * step[pending]
            good = np.max(np.abs(_design(trial) @ coeff - ya[pending]), axis=1) < ea[pending]
            accepted[pending[good]] = trial[good]
            pending = pending[~good]
            if not len(pending):
                break
        x[active] = accepted
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a study in reference coordinates without modifying the input."""
    out = study.copy()
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    coeff = _fit_forward(ref, raw)
    out['y_obs_raw'] = _invert(study['y_obs_raw'], coeff)
    if study.get('reference_tracer_pulses') is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coeff))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in study['reference_tracer_pulses']
        ]
    calibration = sweeps.copy()
    calibration['raw_rosette'] = sweeps['ref_rosette'].copy()
    banks = sweeps['single_bank_sweeps']
    calibration['single_bank_sweeps'] = dict(banks, raw_readout=banks['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = calibration
    return out
