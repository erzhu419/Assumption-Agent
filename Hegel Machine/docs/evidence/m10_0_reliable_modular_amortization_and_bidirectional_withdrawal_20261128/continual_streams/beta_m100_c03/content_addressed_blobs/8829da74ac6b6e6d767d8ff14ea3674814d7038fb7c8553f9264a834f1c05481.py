"""Robust calibration of the coupled four-channel rosette."""

import numpy as np


# The mixed and single-axis sweeps identify two reciprocal channel pairs.
_PARTNER = np.array([2, 3, 0, 1])
_DIAGONAL = np.arange(4)


def _design(x):
    z = x[:, _PARTNER]
    return np.stack((np.ones_like(x), x, np.sin(2.5 * x), z,
                     1.0 - np.cos(2.0 * z)), axis=-1)


def _fit(ref, raw):
    design = _design(ref)
    # Initialize away from gross EMI, then identify spike rows in forward
    # residual space, not according to the size of the reference excitation.
    center = np.median(raw, axis=0)
    scale = np.median(np.abs(raw - center), axis=0)
    keep = np.all(np.abs(raw - center) <= 8 * scale, axis=1)

    def fit_rows(mask):
        return np.array([np.linalg.lstsq(design[mask, j], raw[mask, j],
                                         rcond=None)[0] for j in range(4)])

    for _ in range(8):
        coeff = fit_rows(keep)
        err = np.linalg.norm(np.sum(design * coeff, axis=-1) - raw, axis=1)
        new_keep = err <= 6 * max(np.median(err), 1e-12)
        if np.array_equal(new_keep, keep):
            return coeff
        keep = new_keep
    return fit_rows(keep)


def _inverse(raw, coeff):
    data = np.asarray(raw, dtype=float)
    y = data.reshape(-1, 4)
    offset, linear, sine, coupling, cosine = coeff.T
    x = (y - offset) / linear

    def residual(x):
        z = x[:, _PARTNER]
        return offset + linear * x + sine * np.sin(2.5 * x) + coupling * z + cosine * (1.0 - np.cos(2.0 * z)) - y

    r = residual(x)
    for _ in range(800):
        error = np.max(np.abs(r), axis=1)
        todo = error > 1e-12 * (1 + np.max(np.abs(y), axis=1))
        if not np.any(todo):
            return x.reshape(data.shape)
        u, v = x[todo], r[todo]
        jac = np.zeros((len(u), 4, 4))
        jac[:, _DIAGONAL, _DIAGONAL] = linear + 2.5 * sine * np.cos(2.5 * u)
        jac[:, _DIAGONAL, _PARTNER] = coupling + 2.0 * cosine * np.sin(2.0 * u[:, _PARTNER])
        trial = u - np.linalg.solve(jac, v[..., None])[..., 0]
        trial_r = (offset + linear * trial + sine * np.sin(2.5 * trial)
                   + coupling * trial[:, _PARTNER] + cosine * (1.0 - np.cos(2.0 * trial[:, _PARTNER])) - y[todo])
        # Use the diagonally scaled fixed-point step when Newton overshoots.
        worse = np.max(np.abs(trial_r), axis=1) > 0.9 * error[todo]
        trial[worse] = u[worse] - v[worse] / linear
        x[todo] = trial
        r = residual(x)
    raise RuntimeError('Rosette inversion did not converge')


def calibrate_upstream(study: dict) -> dict:
    """Convert observations and tracer readouts to reference coordinates."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    coeff = None if np.array_equal(raw, ref) else _fit(ref, raw)

    def calibrate(readout):
        return np.asarray(readout).copy() if coeff is None else _inverse(readout, coeff)

    calibrated = dict(study)
    calibrated['y_obs_raw'] = calibrate(study['y_obs_raw'])
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=calibrate(pulse['y_pulse_raw']))
            if pulse is not None and pulse['y_pulse_raw'] is not None else pulse
            for pulse in pulses
        ]
    banks = sweeps['single_bank_sweeps']
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(banks, raw_readout=np.asarray(banks['ref_signal']).copy()),
    )
    return calibrated
