"""Fit and invert the coupled four-channel rosette response for each study."""

import numpy as np


_I, _J = np.triu_indices(4, k=1)
_EYE = np.eye(4)


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x**3, x[:, _I] * x[:, _J]))


def _fit(ref, raw):
    design = _basis(ref)
    # An initial amplitude screen removes gross EMI before it can pull the fit.
    radius = np.linalg.norm(raw - np.median(raw, axis=0), axis=1)
    use = radius <= 6 * np.median(radius)
    if np.count_nonzero(use) < design.shape[1]:
        use = np.ones(len(raw), dtype=bool)
    coef = np.linalg.lstsq(design[use], raw[use], rcond=None)[0]
    # Downweight whole rows: a burst can contaminate several banks together.
    for _ in range(12):
        error = np.linalg.norm(raw - design @ coef, axis=1)
        scale = max(2.5 * np.median(error), 1e-12)
        w = 1 / np.hypot(1, error / scale)
        coef = np.linalg.lstsq(design * w[:, None], raw * w[:, None], rcond=None)[0]
    return coef


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        return raw.copy()
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    for _ in range(36):
        residual = _basis(x) @ coef - y
        error = np.max(np.abs(residual), axis=1)
        active = error > 1e-12 * (1 + np.max(np.abs(y), axis=1))
        if not np.any(active):
            break
        z = x[active]
        pairs = (z[:, _I, None] * _EYE[_J] + z[:, _J, None] * _EYE[_I])
        # Jacobian axes are (row, output channel, input channel).
        jac = (coef[1:5].T + 3 * z[:, None, :]**2 * coef[5:9].T
               + np.einsum('npi,po->noi', pairs, coef[9:]))
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        trial = z - step
        best = error[active]
        for _ in range(18):
            worse = np.max(np.abs(_basis(trial) @ coef - y[active]), axis=1) >= best
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        x[active] = trial
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return the study in canonical readout coordinates without mutating it."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    normalized = dict(sweeps, raw_rosette=ref.copy())
    if 'single_bank_sweeps' in sweeps:
        single = sweeps['single_bank_sweeps']
        normalized['single_bank_sweeps'] = dict(single, raw_readout=np.copy(single['ref_signal']))
    result = dict(study, sensor_calibration_sweeps=normalized,
                  y_obs_raw=_invert(study['y_obs_raw'], coef))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return result
